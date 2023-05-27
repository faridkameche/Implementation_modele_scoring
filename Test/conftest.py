import pytest

from P7_Flask_API_ import read_data


@pytest.fixture
def client():
    app = read_data()
    with app.test_client() as client:
        yield client