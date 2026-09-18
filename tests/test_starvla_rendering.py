"""Renderer cleanup must not corrupt a different simulator's current context."""

from types import SimpleNamespace

import pytest

from oat.starvla_heading import rendering

from oat.starvla_heading.rendering import (
    _configure_egl_module, _cuda_device_query, _free_egl_context, _free_mujoco_context, _physical_cuda_device,
)


def mapping_module(cuda_ordinals):
    devices = [SimpleNamespace(cuda_ordinal=ordinal) for ordinal in cuda_ordinals]
    selected = []

    def display(platform, device, attributes):
        selected.append(device)
        return "display"

    egl = SimpleNamespace(
        eglQueryDevicesEXT=lambda: devices,
        eglGetPlatformDisplayEXT=display,
        eglGetError=lambda: 0,
        eglInitialize=lambda *args: True,
        EGL_PLATFORM_DEVICE_EXT="device", EGL_NO_DISPLAY=None, EGL_SUCCESS=0,
    )
    return SimpleNamespace(EGL=egl, EGL_DISPLAY=None), selected


def test_cuda_gpu_uses_device_attribute_instead_of_egl_enumeration_position():
    # The failed run's actual enumeration: GPU 7 is EGL device 4, not device 7.
    module, selected = mapping_module([3, 2, 1, 0, 7, 6, 5, 4])
    mapping = _configure_egl_module(module, 7, lambda device: device.cuda_ordinal)
    assert mapping["cuda_device_id"] == 7 and mapping["egl_device_id"] == 4
    assert not selected  # Resolving a GPU must not initialize a display.
    assert module.create_initialized_egl_device_display(device_id=7) == "display"
    assert selected[0].cuda_ordinal == 7
    assert _configure_egl_module(module, 7) == mapping
    with pytest.raises(RuntimeError, match="cannot change"):
        _configure_egl_module(module, 6)
    with pytest.raises(RuntimeError, match="requested CUDA GPU 4"):
        module.create_initialized_egl_device_display(device_id=4)


@pytest.mark.parametrize("ordinals", [[0, 1, None], [None, None], [7, 7]])
def test_missing_or_ambiguous_cuda_mapping_fails_without_initializing_gpu(ordinals):
    module, selected = mapping_module(ordinals)
    with pytest.raises(RuntimeError, match="exactly one EGL device"):
        _configure_egl_module(module, 7, lambda device: device.cuda_ordinal)
    assert not selected
    assert not hasattr(module, "_starvla_cuda_mapping")


def test_mapping_cannot_be_installed_after_a_display_already_exists():
    module, selected = mapping_module([7])
    module.EGL_DISPLAY = "existing display"
    with pytest.raises(RuntimeError, match="before creating any EGL display"):
        _configure_egl_module(module, 7, lambda device: device.cuda_ordinal)
    assert not selected


def test_default_rendering_gpu_uses_first_visible_cuda_ordinal(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7,6")
    assert _physical_cuda_device(-1) == 7
    assert _physical_cuda_device(6) == 6
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert _physical_cuda_device(-1) == 0
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-some-uuid")
    with pytest.raises(ValueError, match="integer CUDA_VISIBLE_DEVICES"):
        _physical_cuda_device(-1)



class FakeEGL:
    EGL_NO_CONTEXT = None
    EGL_NO_SURFACE = None
    EGL_DRAW = "draw"
    EGL_READ = "read"

    def __init__(self, current):
        self.current = current
        self.destroyed = []
        self.bound = []

    def eglGetCurrentContext(self):
        return self.current

    def eglGetCurrentDisplay(self):
        return "display"

    def eglGetCurrentSurface(self, kind):
        return kind

    def eglMakeCurrent(self, display, draw, read, context):
        self.bound.append((display, draw, read, context))
        self.current = context
        return True

    def eglDestroyContext(self, display, context):
        assert self.current is not context
        self.destroyed.append(context.address)
        return True

    def eglReleaseThread(self):
        raise AssertionError("Cleanup must not release an unrelated EGL context")


class FakeGLContext:
    def __init__(self, handle, module):
        self._context = handle
        self.module = module

    def make_current(self):
        self.module.EGL.eglMakeCurrent("display", None, None, self._context)

    def free(self):
        _free_egl_context(self, self.module)


def setup_contexts(current="new"):
    old, new = SimpleNamespace(address=1), SimpleNamespace(address=2)
    egl = FakeEGL({"old": old, "new": new, "none": None}[current])
    module = SimpleNamespace(EGL=egl, EGL_DISPLAY="display")
    return egl, FakeGLContext(old, module), old, new


def test_collect_old_renderer_preserves_new_context_and_frees_correct_objects():
    egl, gl_context, old, new = setup_contexts()
    freed = []

    def free_buffers():
        assert egl.current is old, "MuJoCo GL objects must be freed in their owning context"
        freed.append(True)

    renderer = SimpleNamespace(gl_ctx=gl_context, con=SimpleNamespace(free=free_buffers), scn=object())
    _free_mujoco_context(renderer, egl)
    assert egl.current is new
    assert egl.bound[-1] == ("display", "draw", "read", new)
    assert egl.destroyed == [old.address]
    assert freed == [True]
    assert renderer.con is None and renderer.gl_ctx is None and renderer.scn is None
    _free_mujoco_context(renderer, egl)
    assert freed == [True] and egl.destroyed == [old.address]


@pytest.mark.parametrize("current", ["old", "new", "none"])
def test_failed_mujoco_initialization_still_destroys_egl_context(current):
    egl, gl_context, old, new = setup_contexts(current)
    renderer = SimpleNamespace(gl_ctx=gl_context)  # MjrContext raised before `con` assignment.
    _free_mujoco_context(renderer, egl)
    assert egl.destroyed == [old.address]
    assert egl.current is (new if current == "new" else None)
    _free_mujoco_context(renderer, egl)
    assert egl.destroyed == [old.address]


def test_standalone_egl_cleanup_preserves_another_current_context():
    egl, gl_context, old, new = setup_contexts()
    gl_context.free()
    assert egl.current is new
    assert not egl.bound
    assert egl.destroyed == [old.address]
    gl_context.free()
    assert egl.destroyed == [old.address]


def test_cleanup_restores_new_context_even_when_freeing_buffers_raises():
    egl, gl_context, old, new = setup_contexts()

    def fail():
        raise RuntimeError("buffer cleanup failure")

    with pytest.raises(RuntimeError, match="buffer cleanup failure"):
        _free_mujoco_context(SimpleNamespace(gl_ctx=gl_context, con=SimpleNamespace(free=fail)), egl)
    assert egl.current is new
    assert egl.destroyed == [old.address]


def test_cleanup_tolerates_failure_before_egl_context_exists():
    egl = FakeEGL(None)
    _free_mujoco_context(SimpleNamespace(), egl)
    _free_egl_context(SimpleNamespace(), SimpleNamespace(EGL=egl))
    assert not egl.destroyed and egl.current is None


def test_cuda_attribute_is_queried_only_when_device_advertises_exact_extension(monkeypatch):
    queried = []

    def query_attribute(device, attribute, output):
        assert attribute == 0x323A
        queried.append(device)
        output._obj.value = 7
        return True

    procedures = {
        "eglQueryDeviceStringEXT": lambda device, attribute: device,
        "eglQueryDeviceAttribEXT": query_attribute,
    }
    egl = SimpleNamespace(eglGetProcAddress=procedures.get, EGLDeviceEXT=object,
                          EGLint=int, EGLBoolean=bool, EGL_EXTENSIONS=123)
    monkeypatch.setattr(rendering.ctypes, "CFUNCTYPE", lambda *signature: lambda address: address)
    query = _cuda_device_query(egl)
    assert query(b"EGL_EXT_device_drm") is None
    assert query(b"EGL_NV_device_cuda_unrelated") is None
    assert query(None) is None
    assert not queried
    advertised = b"EGL_EXT_device_drm EGL_NV_device_cuda"
    assert query(advertised) == 7
    assert queried == [advertised]


def test_unavailable_cuda_device_query_rejects_instead_of_guessing_gpu():
    with pytest.raises(RuntimeError, match="queries are unavailable"):
        _cuda_device_query(SimpleNamespace(eglGetProcAddress=lambda name: None))
