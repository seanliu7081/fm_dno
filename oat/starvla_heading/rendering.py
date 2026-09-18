"""Select the intended CUDA GPU and keep EGL cleanup inside its owning context.

EGL device enumeration can differ from CUDA device enumeration. Robosuite 1.4
indexes the former using the latter, silently selecting another GPU. Resolve the
physical CUDA device explicitly through EGL_NV_device_cuda before initialization.

Simulator reset can leave an old renderer for Python's cyclic garbage collector.
Its finalizer may then run after another renderer has become current. Upstream
cleanup frees MuJoCo GL objects in whichever context happens to be current and
releases the EGL thread even when destroying an unrelated context. Both actions
can invalidate that newer renderer. Install these process-local fixes before
creating a simulator; installed dependencies are left untouched.
"""

from contextlib import contextmanager
import ctypes
import os


def _cuda_device_query(egl):
    """Return a query that only reads CUDA ordinals from supporting EGL devices."""
    query_string_address = egl.eglGetProcAddress("eglQueryDeviceStringEXT")
    query_attribute_address = egl.eglGetProcAddress("eglQueryDeviceAttribEXT")
    if not query_string_address or not query_attribute_address:
        raise RuntimeError("EGL device queries are unavailable; refusing to guess a CUDA GPU")
    query_string = ctypes.CFUNCTYPE(ctypes.c_char_p, egl.EGLDeviceEXT, egl.EGLint)(query_string_address)
    query_attribute = ctypes.CFUNCTYPE(
        egl.EGLBoolean, egl.EGLDeviceEXT, egl.EGLint, ctypes.POINTER(ctypes.c_ssize_t)
    )(query_attribute_address)

    def query(device):
        extensions = query_string(device, egl.EGL_EXTENSIONS)
        if not extensions or b"EGL_NV_device_cuda" not in extensions.split():
            return None
        cuda_device = ctypes.c_ssize_t(-1)
        if not query_attribute(device, 0x323A, ctypes.byref(cuda_device)):  # EGL_CUDA_DEVICE_NV
            raise RuntimeError("EGL_NV_device_cuda query failed; refusing to guess a CUDA GPU")
        return cuda_device.value

    return query


def _configure_egl_module(module, cuda_device_id, query_cuda_device=None):
    """Resolve a physical CUDA ordinal and bind robosuite to that EGL device."""
    if isinstance(cuda_device_id, bool) or not isinstance(cuda_device_id, int) or cuda_device_id < 0:
        raise ValueError("Rendering requires an explicit nonnegative physical CUDA device ordinal")
    previous = getattr(module, "_starvla_cuda_mapping", None)
    if previous is not None:
        if previous["cuda_device_id"] != cuda_device_id:
            raise RuntimeError("A simulator worker cannot change its configured CUDA rendering GPU")
        return dict(previous)
    if module.EGL_DISPLAY is not None:
        raise RuntimeError("Configure the CUDA rendering GPU before creating any EGL display")
    egl = module.EGL
    query_cuda_device = query_cuda_device or _cuda_device_query(egl)
    mappings, matches = [], []
    for index, device in enumerate(egl.eglQueryDevicesEXT()):
        ordinal = query_cuda_device(device)
        mappings.append({"egl_device_id": index, "cuda_device_id": ordinal})
        if ordinal == cuda_device_id:
            matches.append((index, device))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one EGL device for CUDA GPU {cuda_device_id}; "
                           f"found {len(matches)}. EGL/CUDA mapping: {mappings}")
    egl_device_id, selected_device = matches[0]

    def create_display(device_id=0):
        if device_id not in (-1, cuda_device_id):
            raise RuntimeError(f"Renderer requested CUDA GPU {device_id}, configured GPU is {cuda_device_id}")
        display = egl.eglGetPlatformDisplayEXT(egl.EGL_PLATFORM_DEVICE_EXT, selected_device, None)
        if display == egl.EGL_NO_DISPLAY or egl.eglGetError() != egl.EGL_SUCCESS:
            raise RuntimeError(f"Could not obtain EGL display for CUDA GPU {cuda_device_id}")
        if not egl.eglInitialize(display, None, None) or egl.eglGetError() != egl.EGL_SUCCESS:
            raise RuntimeError(f"Could not initialize EGL display for CUDA GPU {cuda_device_id}")
        return display

    module.create_initialized_egl_device_display = create_display
    mapping = {"cuda_device_id": cuda_device_id, "egl_device_id": egl_device_id,
               "mapping_source": "EGL_NV_device_cuda", "available_devices": mappings}
    module._starvla_cuda_mapping = mapping
    return dict(mapping)


def _physical_cuda_device(cuda_device_id):
    """Resolve the existing -1 default without assuming EGL enumeration order."""
    if cuda_device_id != -1:
        return cuda_device_id
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return 0
    first = visible.split(",")[0].strip()
    if not first.isdigit():
        raise ValueError("Default rendering GPU requires integer CUDA_VISIBLE_DEVICES ordinals")
    return int(first)


def configure_robosuite_egl(cuda_device_id):
    """Select a physical CUDA GPU before the worker's first simulator is created.

    `cuda_device_id` is a CUDA device ordinal, not an EGL enumeration index and
    not a rank-local index into CUDA_VISIBLE_DEVICES. The periodic evaluator uses
    an unreordered CUDA_VISIBLE_DEVICES list to keep physical ordinals explicit.
    -1 selects the first visible physical CUDA device, or 0 when visibility is
    unrestricted. The returned mapping is suitable for logging diagnostics.
    """
    from robosuite.renderers.context import egl_context
    from robosuite.utils import binding_utils

    if binding_utils.GLContext is not egl_context.EGLGLContext:
        raise RuntimeError("StarVLA evaluation requires robosuite's EGL renderer")
    return _configure_egl_module(egl_context, _physical_cuda_device(cuda_device_id))


def _same_context(left, right):
    return bool(left) and bool(right) and left.address == right.address


@contextmanager
def _own_context(gl_context, egl):
    """Use an owner's context temporarily, preserving another active renderer."""
    previous = egl.eglGetCurrentContext()
    restore = bool(previous) and not _same_context(previous, gl_context._context)
    if restore:
        display = egl.eglGetCurrentDisplay()
        draw = egl.eglGetCurrentSurface(egl.EGL_DRAW)
        read = egl.eglGetCurrentSurface(egl.EGL_READ)
    try:
        gl_context.make_current()
        yield
    finally:
        try:
            gl_context.free()
        finally:
            if restore and not egl.eglMakeCurrent(display, draw, read, previous):
                raise RuntimeError("Could not restore EGL context after renderer cleanup")


def _free_egl_context(context, module):
    """Destroy this context without releasing a different current EGL context."""
    handle = getattr(context, "_context", None)
    if not handle:
        return
    egl = module.EGL
    if _same_context(handle, egl.eglGetCurrentContext()):
        if not egl.eglMakeCurrent(module.EGL_DISPLAY, egl.EGL_NO_SURFACE,
                                 egl.EGL_NO_SURFACE, egl.EGL_NO_CONTEXT):
            raise RuntimeError("Could not unbind EGL context during renderer cleanup")
    if not egl.eglDestroyContext(module.EGL_DISPLAY, handle):
        raise RuntimeError("Could not destroy EGL context during renderer cleanup")
    context._context = None
    # eglReleaseThread() also unbinds any OTHER context currently on this thread.
    # Destroying our context is sufficient; EGL releases thread state at exit.


def _free_mujoco_context(renderer, egl):
    """Handle both complete renderers and failed MjrContext construction."""
    gl_context = getattr(renderer, "gl_ctx", None)
    con = getattr(renderer, "con", None)
    if gl_context is not None:
        if con is not None and getattr(gl_context, "_context", None):
            with _own_context(gl_context, egl):
                con.free()
                renderer.con = None
        else:
            # MjrContext can fail before assigning `con`, while EGL already owns
            # a context. Its cleanup must still happen in that case.
            gl_context.free()
        renderer.gl_ctx = None
    for name in ("scn", "cam", "vopt", "pert"):
        if hasattr(renderer, name):
            setattr(renderer, name, None)


def install_robosuite_egl_cleanup():
    """Install idempotent cleanup fixes in this simulator worker only."""
    from robosuite.renderers.context import egl_context
    from robosuite.utils import binding_utils

    if binding_utils.GLContext is not egl_context.EGLGLContext:
        raise RuntimeError("StarVLA evaluation requires robosuite's EGL renderer")
    renderer = binding_utils.MjRenderContext
    if getattr(renderer, "_starvla_egl_cleanup", False):
        return

    def free_context(self):
        _free_egl_context(self, egl_context)

    def free_renderer(self):
        _free_mujoco_context(self, egl_context.EGL)

    egl_context.EGLGLContext.free = free_context
    renderer.__del__ = free_renderer
    renderer._starvla_egl_cleanup = True
