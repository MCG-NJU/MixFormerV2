import os
import gradio as gr


def javascript_html():
    # Ensure localization is in `window` before scripts
    # head = '<script type="text/javascript">window.localization = {}</script>\n'
    head = ''
    head += f'<script type="text/javascript" src="file=javascript/bboxHint.js?{os.path.getmtime("javascript/bboxHint.js")}"></script>\n'

    return head


GradioTemplateResponseOriginal = gr.routes.templates.TemplateResponse
def reload_javascript():
    js = javascript_html()

    def template_response(*args, **kwargs):
        res = GradioTemplateResponseOriginal(*args, **kwargs)
        res.body = res.body.replace(b'</head>', f'{js}</head>'.encode("utf8"))
        res.init_headers()
        return res

    gr.routes.templates.TemplateResponse = template_response
