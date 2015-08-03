/* To calculate the velocity in optical flow problem */



#include <base/quadrature_lib.h>
#include <base/function.h>
#include <base/function_time.h>
#include <base/logstream.h>
#include <lac/vector.h>
#include <lac/full_matrix.h>
#include <lac/sparse_matrix.h>
#include <lac/solver_cg.h>
#include <lac/sparse_direct.h>
#include <lac/precondition.h>
#include <lac/constraint_matrix.h>
#include <grid/tria.h>
#include <grid/grid_generator.h>
#include <grid/grid_refinement.h>
#include <grid/tria_accessor.h>
#include <grid/tria_iterator.h>
#include <grid/tria_boundary_lib.h>
#include <dofs/dof_handler.h>
#include <dofs/dof_accessor.h>
#include <dofs/dof_tools.h>
#include <fe/fe_values.h>
#include <numerics/vectors.h>
#include <numerics/matrices.h>
#include <numerics/data_out.h>
#include <numerics/error_estimator.h>


#include <fe/fe_system.h>

#include <fe/fe_q.h>


#include <fstream>
using std::ofstream;

#include <iostream>
using std::cerr;
using std::endl;

#include"optical.h"
//#include"image.h"

using namespace dealii;


int main ()
{
   int i;
   double smpar[10],err[10];
   for( i = 0; i <10; i++)
   {
     double k;
     if(i==0)
       smpar[i] = 2;
     else
       smpar[i] = 10*(i);
      k=smpar[i];
     try
     {
        deallog.depth_console (0);
        unsigned int degree = 1;
      
        bool dirichlet = false;
        OFProblem<2> of_problem(degree,
                              k,
                              dirichlet,
                              OFProblem<2>::global_refinement);
        err[i] = of_problem.run (i);
        double g_error = of_problem.advec_err();
        std::cout<<"The Advection Error is:"<<g_error<<std::endl;
     }
     catch (std::exception &exc)
     {
        std::cerr << std::endl << std::endl
		          << "----------------------------------------------------"
		          << std::endl;
        std::cerr << "Exception on processing: " << std::endl
		          << exc.what() << std::endl
		          << "Aborting!" << std::endl
		          << "----------------------------------------------------"
		          << std::endl;
      
        return 1;
     }
     catch (...)
     {
        std::cerr << std::endl << std::endl
		          << "----------------------------------------------------"
		          << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
        return 1;
     }
   }
  // Writing data into 2 files.
   ofstream smpar_data; 
   smpar_data.open("Smoothing_Term.txt"); // opens the file
   if( !smpar_data ) { // file couldn't be opened
      cerr << "Error: file could not be opened" << endl;
      exit(1);
   }

  for (i=0; i<10; ++i)
      smpar_data << smpar[i] << endl;
  smpar_data.close();
  
   ofstream error_data; 
   error_data.open("L2_Error.txt"); // opens the file
   if( !error_data) { // file couldn't be opened
      cerr << "Error: file could not be opened" << endl;
      exit(1);
   }

  for (i=0; i<10; ++i)
      error_data << err[i] << endl;
 error_data.close();


   return 0;
}
