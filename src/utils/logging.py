import requests
import time

app_log = []
clear_log_after = 50


def get_log(tags=["INFO"]):
    log = list(
        map(
            lambda l: l["message"],
            filter(lambda l: bool(set(l["tags"]) & set(tags)), app_log),
        )
    )

    log.reverse()
    return log


def k_error(message):
    k_log(message, ["INFO, ERROR"])


def k_log(message, tags=["INFO"]):
    current_time = time.strftime("%H:%M:%S", time.localtime())
    if len(app_log) > clear_log_after:
        app_log.clear()
        print(f"log was cleared because it reached max size ({clear_log_after})")
    app_log.append({"tags": tags, "message": f"[{current_time}]: {message}"})
    print(message)
