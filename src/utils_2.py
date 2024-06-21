from model_ecg.net import Encoder, Decoder

def encode_templates(class_template_mapping):
    encoder = Encoder()
    z_class_template_mapping = {}
    for key, value in class_template_mapping.items():
        if key not in z_class_template_mapping.keys():
            z_class_template_mapping[key] = []
        for o in value:
            z = encoder.predict(o)
            z_class_template_mapping[key].append(z)
    return z_class_template_mapping