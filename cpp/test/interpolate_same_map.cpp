#include <cstddef>
#include <cudolfinx/fem/CUDACoefficient.h>
#include <dolfinx.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/mesh/generation.h>
#include <memory>
#include <mpi.h>
#include <stdexcept>
#include <chrono>

using std::chrono::duration;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

template<typename T>
void printVec(const T& vec) {
  for (auto v : vec) {
    std::cout << v << " ";
  }
  std::cout << std::endl;
}

template<typename T, typename U>
void allClose(const T& v1, const U& v2) {
  assert(v1.size() == v2.size());
  for (std::size_t i = 0; i < v1.size(); i++) {
    assert(std::abs((v1[i] - v2[i])/(v1[i])) < 1e-13);
  }
}

using T = double;
using namespace dolfinx;

int main(int argc, char* argv[]) {
  init_logging(argc, argv);
  MPI_Init(&argc, &argv);
  
  CUdevice cuDevice = 0;
  CUcontext cuContext;
  const char * cuda_err_description;

  cuInit(0);
  int device_count;
  CUresult cuda_err = cuDeviceGetCount(&device_count);
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error("cuDeviceGetCount failed with " +
                             std::string(cuda_err_description) + " at " +
                             std::string(__FILE__) + ":" +
                             std::to_string(__LINE__));
  }
  std::cout << "No. of devices: " << device_count << std::endl;

  cuCtxCreate(&cuContext, 0, cuDevice);
  const int num_cells = 30;
  const T lower = 0.;
  const T upper = 1.;
  const int p_order = 5;

  auto element = basix::create_element<T>(
      basix::element::family::P, basix::cell::type::tetrahedron, p_order,
      basix::element::lagrange_variant::equispaced,
      basix::element::dpc_variant::unset, false);

  auto element_from = basix::create_element<T>(
      basix::element::family::P, basix::cell::type::tetrahedron, 6,
      basix::element::lagrange_variant::equispaced,
      basix::element::dpc_variant::unset, false);

  auto e0 = std::make_shared<fem::FiniteElement<T>>(element);
  auto e1 = std::make_shared<fem::FiniteElement<T>>(element_from);

  const auto mesh = std::make_shared<mesh::Mesh<T>>(mesh::create_box(
      MPI_COMM_WORLD, {{{lower, lower, lower}, {upper, upper, upper}}},
      {num_cells, num_cells, num_cells}, mesh::CellType::tetrahedron));

  auto V = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace<T>(mesh, e0));
  auto V_from = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace<T>(mesh, e1));
  auto f = std::make_shared<fem::Function<T>>(V);
  auto f_true = std::make_shared<fem::Function<T>>(V);
  auto f_from = std::make_shared<fem::Function<T>>(V_from);

  /* Interpolate the function to interpolate FROM */
  f_from->interpolate(
      [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
      {
        std::vector<T> f;
        for (std::size_t p = 0; p < x.extent(1); ++p)
        {
          f.push_back(1 + 0.10*x(0,p)*x(0,p) + 0.2*x(1,p)*x(1,p) + 0.3*x(2,p)*x(2,p));
        }

        return {f, {f.size()}};
      });

  /* Reference version */
  const std::size_t ITER = 5;
  auto t1 = high_resolution_clock::now();
  for (std::size_t i = 0; i < ITER; i++)
    f_true->interpolate(*f_from);
  auto t2 = high_resolution_clock::now();
  duration<double, std::milli> ms = t2 - t1;
  std::cout << "Reference implementation: " << ms.count()/(double)ITER << " ms" << std::endl;


  t1 = high_resolution_clock::now();
  auto coeffs = dolfinx::fem::CUDACoefficient<double>(f) ;
  t2 = high_resolution_clock::now();
  ms = t2 - t1;
  //std::cout << "CUDA initialization: " << ms.count() << " ms" << std::endl;


  /* GPU version */
  auto coeffs_from = dolfinx::fem::CUDACoefficient<double>(f_from);
  coeffs.interpolate(coeffs_from);
  t1 = high_resolution_clock::now();
  for (std::size_t i = 0; i < ITER; i++) {
    coeffs.interpolate(coeffs_from);
  }
  t2 = high_resolution_clock::now();
  ms = t2 - t1;
  std::cout << "GPU implementation: " << ms.count()/(double)ITER << " ms" << std::endl;
  allClose(f_true->x()->array(), coeffs.values());



  return 0;
}

