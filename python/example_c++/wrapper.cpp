#include <Python.h>

class TestFact{
    public:
    TestFact(){};
    ~TestFact(){};
    int fact(int n);
};

int TestFact::fact(int n)
{
  if (n <= 1)
    return 1;
  else
    return n * (n - 1);
}

int fact(int n)
{
    TestFact t;
    return t.fact(n);
}

PyObject* wrap_fact(PyObject* self, PyObject* args)
{
  int n, result;

  if (! PyArg_ParseTuple(args, "i:fact", &n))
    return NULL;
  result = fact(n);
  return Py_BuildValue("i", result);
}

static PyMethodDef exampleMethods[] =
{
  {"fact", wrap_fact, METH_VARARGS, "Caculate N!"},
  {NULL, NULL}
};

extern "C"
void initexample()
{
  PyObject* m;
  m = Py_InitModule("example", exampleMethods);
}
