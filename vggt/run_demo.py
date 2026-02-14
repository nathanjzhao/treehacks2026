"""Wrapper: force HTTPS scheme in Gradio's ASGI scope for Modal proxy compat."""
import sys, os
sys.path.insert(0, "/opt/vggt")
os.chdir("/opt/vggt")

# Monkey-patch Gradio's ASGI app to report https scheme,
# so generated file URLs use https:// instead of http://
import gradio.routes

_orig_call = gradio.routes.App.__call__

async def _https_call(self, scope, receive, send):
    scope["scheme"] = "https"
    return await _orig_call(self, scope, receive, send)

gradio.routes.App.__call__ = _https_call

# Run the demo
exec(open("demo_gradio.py").read())
