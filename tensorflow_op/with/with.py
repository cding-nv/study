
class sample:
    def __enter__(self):
        print("in enter")
        return "Foo"
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("in exit")
def get_sample():
    return sample()
with get_sample() as sample:
    print("sample: ", sample)
