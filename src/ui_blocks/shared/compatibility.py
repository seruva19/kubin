def negative_prompt_classes():
    return [
        "unsupported_20",
    ]


def send_outpaint_btn_classes():
    return [
        "unsupported_20",
    ]


def send_mix_btn_classes():
    return [
        "unsupported_20",
        "unsupported_30",
        "unsupported_d30",
        "unsupported_31",
    ]


def prior_block_classes():
    return [
        "unsupported_20",
        "unsupported_d30",
        "unsupported_30",
        "unsupported_31",
    ]


def batch_size_classes():
    return [
        "unsupported_20",
    ]


def ckpt_selector20_classes():
    return [
        "unsupported_21",
        "unsupported_d21",
        "unsupported_22",
    ]


def ckpt_selector21_classes():
    return [
        "unsupported_20",
        "unsupported_d21",
        "unsupported_22",
    ]


def ckpt_selector21d_classes():
    return [
        "unsupported_20",
        "unsupported_21",
        "unsupported_22",
        "unsupported_d22",
    ]


def ckpt_selector22_classes():
    return [
        "unsupported_20",
        "unsupported_21",
        "unsupported_d21",
    ]


def ext_availability_classes(ext_augment):
    classes = []
    supports_pipeline_model = ext_augment.get(
        "supports",
        [
            "native-kd31",
            "diffusers-kd30",
            "native-kd30",
            "diffusers-kd22",
            "diffusers-kd21",
            "native-kd21",
            "native-kd20",
        ],
    )
    if "native-kd20" not in supports_pipeline_model:
        classes.append("unsupported_20")

    if "native-kd21" not in supports_pipeline_model:
        classes.append("unsupported_21")

    if "native-kd30" not in supports_pipeline_model:
        classes.append("unsupported_30")

    if "native-kd31" not in supports_pipeline_model:
        classes.append("unsupported_31")

    if "diffusers-kd21" not in supports_pipeline_model:
        classes.append("unsupported_d21")

    if "diffusers-kd22" not in supports_pipeline_model:
        classes.append("unsupported_d22")

    if "diffusers-kd30" not in supports_pipeline_model:
        classes.append("unsupported_d30")

    return classes


def generate_rules():
    return """
        body[class*="pipeline-native"] .diffusers-control {
            display: none;
        }

        body[class*="pipeline-diffusers"] .native-control {
            display: none;
        }

        body:not([class*="pipeline-diffusers-kd22"]) .diffusers-kd22-control {
            display: none;
        }

        body[class*="pipeline-"][class*="-kd20"] [class*="unsupported_20"],
        body[class*="pipeline-native-kd21"] [class*="unsupported_21"],
        body[class*="pipeline-diffusers-kd21"] [class*="unsupported_d21"],
        body[class*="pipeline-native-kd22"] [class*="unsupported_22"],
        body[class*="pipeline-diffusers-kd22"] [class*="unsupported_d22"],
        body[class*="pipeline-native-kd30"] [class*="unsupported_30"],
        body[class*="pipeline-diffusers-kd30"] [class*="unsupported_d30"],
        body[class*="pipeline-native-kd31"] [class*="unsupported_31"] {
            display: none;
        }

        body[class*="pipeline-"][class*="-kd20"] .ui-tabs>.tab-nav>button:nth-child(3),
        body[class*="pipeline-"][class*="-kd20"] .ui-tabs>.tab-nav>button:nth-child(5) {
            display: none;
        }

        body[class*="pipeline-native-kd30"] .ui-tabs>.tab-nav>button:nth-child(2),
        body[class*="pipeline-native-kd30"] .ui-tabs>.tab-nav>button:nth-child(3),
        body[class*="pipeline-native-kd30"] .ui-tabs>.tab-nav>button:nth-child(5) {
            display: none;
        }

        body[class*="pipeline-diffusers-kd30"] .ui-tabs>.tab-nav>button:nth-child(3),
        body[class*="pipeline-diffusers-kd30"] .ui-tabs>.tab-nav>button:nth-child(4),
        body[class*="pipeline-diffusers-kd30"] .ui-tabs>.tab-nav>button:nth-child(5) {
            display: none;
        }

        body[class*="pipeline-native-kd31"] .ui-tabs>.tab-nav>button:nth-child(2),
        body[class*="pipeline-native-kd31"] .ui-tabs>.tab-nav>button:nth-child(3),
        body[class*="pipeline-native-kd31"] .ui-tabs>.tab-nav>button:nth-child(5) {
            display: none;
        }

        body[class*="pipeline-"][class*="-kd20"] .settings-tabs>.tabs>.tab-nav>button:nth-child(2),
        body[class*="pipeline-diffusers-kd21"] .settings-tabs>.tabs>.tab-nav>button:nth-child(2),
        body[class*="pipeline-"][class*="-kd22"] .settings-tabs>.tabs>.tab-nav>button:nth-child(2),
        body[class*="pipeline-"][class*="-kd30"] .settings-tabs>.tabs>.tab-nav>button:nth-child(2),
        body[class*="pipeline-"][class*="-kd31"] .settings-tabs>.tabs>.tab-nav>button:nth-child(2) {
            display: none;
        }

        body:not(.pipeline-diffusers-kd22) .control-net {
            display: none;
        }
        
        body:not(.pipeline-native-kd31) .ip-adapter {
            display: none;
        }
    """


def sampler20_classes():
    return [
        "unsupported_21",
        "unsupported_d21",
        "unsupported_22",
        "unsupported_d22",
        "unsupported_30",
        "unsupported_d30",
        "unsupported_31",
    ]


def sampler21_classes():
    return [
        "unsupported_20",
        "unsupported_d21",
        "unsupported_22",
        "unsupported_30",
        "unsupported_d30",
        "unsupported_31",
    ]


def sampler_diffusers_classes():
    return [
        "unsupported_20",
        "unsupported_21",
        "unsupported_30",
        "unsupported_d30",
        "unsupported_31",
    ]
