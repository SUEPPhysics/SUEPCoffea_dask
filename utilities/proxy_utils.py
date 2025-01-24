import os
import subprocess
import logging

def check_proxy(time_min=100):
    """
    Checks for existence of proxy with at least time_min
    left on it.
    If it's inactive or below time_min, it will regenerate
    it with 140 hours.
    """
    home_base = os.environ["HOME"]
    proxy_name = f"x509up_u{os.getuid()}"
    proxy = os.path.join(home_base, proxy_name)
    os.environ["X509_USER_PROXY"] = proxy
    regenerate_proxy = False
    if not os.path.isfile(proxy):
        logging.warning("--- proxy file does not exist")
        regenerate_proxy = True
    else:
        lifetime = subprocess.check_output(
            ["voms-proxy-info", "--file", proxy, "--timeleft"]
        )
        lifetime = float(lifetime)
        lifetime = lifetime / (60 * 60)
        if lifetime < time_min:
            logging.warning("--- proxy has expired !")
            regenerate_proxy = True

    if regenerate_proxy:
        redone_proxy = False
        while not redone_proxy:
            status = os.system(f"voms-proxy-init -voms cms --hours=140")
            lifetime = 140
            if os.WEXITSTATUS(status) == 0:
                redone_proxy = True

    return proxy, lifetime