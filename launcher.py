import os
import sys
import time
import socket
import threading
import webview


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


def wait_for_server(port, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.connect(('127.0.0.1', port))
                return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            time.sleep(0.5)
    return False


def run_streamlit(port):

    import streamlit.web.bootstrap as bootstrap
    import streamlit.web.cli as stcli

    bootstrap._set_up_signal_handler = lambda *args, **kwargs: None

    app_path = resource_path("app.py")
    sys.argv = [
        "streamlit", "run", app_path,
        "--server.port", str(port),
        "--server.headless", "true",
        "--server.address", "127.0.0.1",
        "--browser.gatherUsageStats", "false",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
    ]
    stcli.main()


if __name__ == "__main__":
    port = get_free_port()
    thread = threading.Thread(target=run_streamlit, args=(port,), daemon=True)
    thread.start()

    if not wait_for_server(port, timeout=60):
        sys.exit("Streamlit server failed to start in time.")

    webview.create_window(
        "Waste Classification System",
        f"http://127.0.0.1:{port}",
        width=1400,
        height=900,
        min_size=(1000, 700)
    )
    webview.start()