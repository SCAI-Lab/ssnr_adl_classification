
def order_templates(templates_loaded, labels_loaded):
    class_template_mapping = {}

    for t, c in zip(templates_loaded, labels_loaded):
        if c not in class_template_mapping.keys():
            class_template_mapping[c] = []
        
        for template in t:
            class_template_mapping[c].append(template)
    return class_template_mapping



