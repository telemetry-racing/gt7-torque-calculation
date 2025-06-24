import base64
import numpy as np
import torque
import json
import numbers
import traceback


def handler(event, context):
    gears = event.get("gears")
    if gears is None:
        return create_error("gears field is missing", context)
    elif not isinstance(gears, list):
        return create_error("gears is not a list", context)
    else:
        for i in gears:
            if not isinstance(i, numbers.Number):
                return create_error("gears should contains only numbers", context)

    gears_speed = event.get("gearsSpeed")
    if gears_speed is None:
        return create_error("gearsSpeed field is missing", context)
    elif not isinstance(gears_speed, list):
        return create_error("gearsSpeed is not a list", context)
    else:
        for i in gears_speed:
            if not isinstance(i, numbers.Number):
                return create_error("gearsSpeed should contains only numbers", context)

    if len(gears) != len(gears_speed):
        return create_error(f"missing values, gears have {len(gears)} elements" +
                            f" and gearsSpeed have {len(gears_speed)} elements", context)

    final_ratio = event.get("finalRatio")
    if final_ratio is None:
        return create_error("finalRatio field is missing", context)
    elif not isinstance(final_ratio, numbers.Number):
        return create_error("finalRatio is not a number", context)

    string = event.get("img")
    if final_ratio is None:
        return create_error("img field is missing, it should be base64 encoded screenshot", context)

    try:
        jpg_original = base64.b64decode(string)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)

        result = torque.optimal_shifting_points(screenshot=jpg_as_np,
                                                gears=gears,  # [2.668, 2.172, 1.686, 1.357, 1.097, 0.916],
                                                gears_max_speed=gears_speed,  # [104, 128, 166, 206, 255, 316],
                                                final_ratio=final_ratio,  # 3.918,
                                                output_function=torque.create_output_web
                                                )
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps(result)
        }
    except Exception:
        traceback.print_exc()
        return create_error("Can't process the screenshot", context)


def create_error(err_message, context):
    return {
        "statusCode": 400,
        "errorMessage": err_message,
        "requestId": context.aws_request_id
    }
