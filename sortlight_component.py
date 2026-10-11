import os
import streamlit.components.v1 as components

_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sortlight_frontend")

_component_func = components.declare_component(
    "sortlight_ui",
    path=_FRONTEND_DIR,
)


def sortlight_ui(
    class_names,
    class_colors=None,
    model_version=None,
    accuracy=None,
    macro_f1=None,
    about_text=None,
    samples=None,
    result=None,
    key=None,
):
    return _component_func(
        classNames=class_names,
        classColors=class_colors or {},
        modelVersion=model_version,
        accuracy=accuracy,
        macroF1=macro_f1,
        aboutText=about_text,
        samples=samples or [],
        result=result,
        default=None,
        key=key,
    )
