import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import matplotlib.pyplot as plt
import onnxruntime as ort
import numpy as np


class Encoder():
    def __init__(self) -> None:
        self.session = ort.InferenceSession("./model_ecg/onnx_encoder.onnx")

    def predict(self, input_array):
        input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        input_type = self.session.get_inputs()[0].type


        arrays_stacked = np.array(input_array)

        # Add an extra dimension to get the shape (n, 400, 1)
        arrays_stacked = np.expand_dims(arrays_stacked, axis=-1)
        if arrays_stacked.ndim == 2:
            arrays_stacked = np.expand_dims(arrays_stacked, axis=0)
        arrays_stacked = arrays_stacked.astype(np.float32)
        output = self.session.run(None, {input_name: arrays_stacked})[0][0]

        return output
    

class Decoder():
    def __init__(self) -> None:
        self.session = ort.InferenceSession("./model_ecg/onnx_decoder.onnx")
    
    def predict(self, z):
        input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        input_type = self.session.get_inputs()[0].type

        z = np.asarray(z, dtype=np.float32)
        z = np.expand_dims(z, axis=0)

        output = self.session.run(None, {input_name: z})[0]
        return output.flatten()
    
    def generate_random(self):
        random_input_float = np.random.rand(1, 25).astype(np.float32)
        return self.predict(random_input_float)