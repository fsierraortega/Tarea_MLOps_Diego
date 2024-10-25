from app import model_pred

new_data = {
            'Major_Axis_Length': 166,
            'Perimeter': 414,
            'Area': 11266,
            'Convex_Area': 11452,
            'Eccentricity': 0.849
}


def test_predict():
    prediction = model_pred(new_data)
    assert prediction == 1
    print(prediction)