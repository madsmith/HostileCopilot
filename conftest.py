# conftest.py
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "dataset(name): dataset selector for parametrization of ping analysis tests",
    )