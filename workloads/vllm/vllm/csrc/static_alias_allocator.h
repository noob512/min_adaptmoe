#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>

// Python C-API entry points implemented by static_alias_allocator.cpp.
//
// They intentionally live in the existing cumem_allocator extension so that
// CUDA driver calls use the same process and primary CUDA context as PyTorch.
// The regular CUDAPluggableAllocator code path does not call any of these
// functions; static-alias allocations are explicit opt-in objects.
PyObject* python_static_alias_create(PyObject* self, PyObject* args);
PyObject* python_static_alias_info(PyObject* self, PyObject* args);
PyObject* python_static_alias_dlpack(PyObject* self, PyObject* args);
PyObject* python_static_alias_synchronize(PyObject* self, PyObject* args);
PyObject* python_static_alias_destroy(PyObject* self, PyObject* args);
