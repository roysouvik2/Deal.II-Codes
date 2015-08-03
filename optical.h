#ifndef __OPTICAL_H__
#define __OPTICAL_H__

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
#include <iostream>



using namespace dealii;

template <int dim>
class OFProblem
{
public:
   enum RefinementMode 
   {
      global_refinement, adaptive_refinement
   };
   OFProblem (unsigned int   degree,
              double         k,
              bool           dirichlet,
              RefinementMode refinement_mode);
   ~OFProblem ();
   double run (double k);
   double advec_err();
private:
   void setup_system ();
   void assemble_system ();
   void solve ();
   void refine_grid ();
  
   double output_results (const unsigned int cycle,double k) const;
   
   Triangulation<dim>   triangulation;
   DoFHandler<dim>      dof_handler;
   
   FESystem<dim>        fe;
   
   ConstraintMatrix     hanging_node_constraints;
   
   SparsityPattern      sparsity_pattern;
   SparseMatrix<double> system_matrix;
   double               k;
   bool                 dirichlet;
   const RefinementMode refinement_mode;
   
   Vector<double>       solution;
   Vector<double>       system_rhs;
   double               time;

};
#endif
