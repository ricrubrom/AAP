import tensorflow as tf
from typing import List


def configure_cuda(memory_growth: bool = True) -> bool:
    """
    Verifica GPUs disponibles y configura TensorFlow para usar CUDA.
    Si memory_growth es True, habilita growth para cada GPU.
    Devuelve True si se detectaron GPUs, False en caso contrario.
    """
    gpus: List[tf.config.PhysicalDevice] = tf.config.list_physical_devices(
        'GPU')
    if not gpus:
        print("No se detectaron GPUs. TensorFlow usará la CPU.")
        return False

    try:
        if memory_growth:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        print("TensorFlow está configurado para usar CUDA (GPU).")
        return True
    except RuntimeError as e:
        # Ocurre si la configuración de GPUs ya se hizo antes de crear contextos
        print(e)
        return True
