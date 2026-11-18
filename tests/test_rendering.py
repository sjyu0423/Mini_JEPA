import numpy as np

from mini_jepa.rendering import render_frame


def test_rendered_image_shape_dtype_and_values() -> None:
    frame = render_frame((0.5, 0.5), ball_radius=0.06)
    assert frame.shape == (64, 64)
    assert frame.dtype == np.uint8
    assert frame.max() == 255
    assert frame.min() == 0
    assert frame.sum() > 0
