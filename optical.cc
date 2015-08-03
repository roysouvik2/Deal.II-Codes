

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

#include "optical.h"
#include "image.h"

using namespace dealii;



template <int dim>
OFProblem<dim>::OFProblem (unsigned int   degree,
                           double         k,
                           bool           dirichlet,
                           RefinementMode refinement_mode)
:
dof_handler (triangulation),
fe (FE_Q<dim>(degree), dim),
k (k),
dirichlet (dirichlet),
refinement_mode (refinement_mode),
time(0)
{}

template <int dim>
OFProblem<dim>::~OFProblem ()
{
   dof_handler.clear ();
}



template <int dim>
void OFProblem<dim>::setup_system ()
{
   dof_handler.distribute_dofs (fe);
   
   hanging_node_constraints.clear ();
   DoFTools::make_hanging_node_constraints (dof_handler,
                                            hanging_node_constraints);
   hanging_node_constraints.close ();
   
   sparsity_pattern.reinit (dof_handler.n_dofs(),
                            dof_handler.n_dofs(),
                            dof_handler.max_couplings_between_dofs());
   DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern);
   
   hanging_node_constraints.condense (sparsity_pattern);
   
   sparsity_pattern.compress();
   
   system_matrix.reinit (sparsity_pattern);
   
   solution.reinit (dof_handler.n_dofs());
   system_rhs.reinit (dof_handler.n_dofs());
}



template <int dim>
void OFProblem<dim>::assemble_system ()
{
   QGauss<dim>  quadrature_formula(fe.degree + 1);
   
   FEValues<dim> fe_values (fe, quadrature_formula,
                            update_values   | update_gradients |
                            update_quadrature_points | update_JxW_values);
   
   const unsigned int   dofs_per_cell = fe.dofs_per_cell;
   const unsigned int   n_q_points    = quadrature_formula.size();
   
   FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
   Vector<double>       cell_rhs (dofs_per_cell);
   
   std::vector<unsigned int> local_dof_indices (dofs_per_cell);
   
   Image<dim> image;
   image.set_time(time);
   
   std::vector<Tensor<1,dim> > image_grad_list (n_q_points, Tensor<1,dim>());
   
   Image_gradt<dim> imaget;
   imaget.set_time(time);
   
   RightHandSide<dim>      right_hand_side;
   right_hand_side.set_time(time);
   std::vector<Vector<double> > rhs_values (n_q_points,
                                            Vector<double>(dim));
   
   std::cout << "Time = " << time << std::endl;
   
   typename DoFHandler<dim>::active_cell_iterator 
   cell = dof_handler.begin_active(),
   endc = dof_handler.end();
   for (; cell!=endc; ++cell)
   {
      cell_matrix = 0;
      cell_rhs = 0;
      
      fe_values.reinit (cell);
      
      right_hand_side.vector_value_list (fe_values.get_quadrature_points(),
                                         rhs_values);
      
      image.gradient_list (fe_values.get_quadrature_points(), image_grad_list);
      
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
         const unsigned int
         component_i = fe.system_to_component_index(i).first;
         
         for (unsigned int j=0; j<dofs_per_cell; ++j)
         {
            const unsigned int
            component_j = fe.system_to_component_index(j).first;
            
            for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
            {
               cell_matrix(i,j)
               +=
               (image_grad_list[q_point][component_i] *
                fe_values.shape_value(i,q_point) *
                image_grad_list[q_point][component_j] *
                fe_values.shape_value(j,q_point)
                +
                ((component_i == component_j) ?
                (k*fe_values.shape_grad(i,q_point) *
                   fe_values.shape_grad(j,q_point))
                :
                0)) *
                fe_values.JxW(q_point);
            }
         }
      }
      
      // compute rhs
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
         const unsigned int
         component_i = fe.system_to_component_index(i).first;
         
         for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
            cell_rhs(i) += fe_values.shape_value(i,q_point) *
			                  rhs_values[q_point](component_i) *
			                  fe_values.JxW(q_point);
      }
      
      // Now add contribution to system matrix and rhs
      cell->get_dof_indices (local_dof_indices);
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
         for (unsigned int j=0; j<dofs_per_cell; ++j)
            system_matrix.add (local_dof_indices[i],
                               local_dof_indices[j],
                               cell_matrix(i,j));
         
         system_rhs(local_dof_indices[i]) += cell_rhs(i);
      }
   }
   
   hanging_node_constraints.condense (system_matrix);
   hanging_node_constraints.condense (system_rhs);
   
   // Apply boundary condition only if dirichlet==true
   if(dirichlet)
   {
      std::map<unsigned int,double> boundary_values;
      VectorTools::interpolate_boundary_values (dof_handler,
                                                0,
                                                ZeroFunction<dim>(dim),
                                                boundary_values);
      MatrixTools::apply_boundary_values (boundary_values,
                                          system_matrix,
                                          solution,
                                          system_rhs);
   }
   
}




template <int dim>
void OFProblem<dim>::solve ()
{   
   // Select linear solver
   int type = 1;
   
   switch(type)
   {
      // Conjugate gradient
      case 1:
      {
         SolverControl  solver_control (5000, 1e-12);
         SolverCG<>     cg (solver_control);
         
         PreconditionSSOR<> preconditioner;
         preconditioner.initialize(system_matrix, 1.2);
         
         cg.solve (system_matrix, 
                   solution, 
                   system_rhs,
                   preconditioner);
         break;
      }
      
      // Direct solver
      case 2:
      {
         SparseDirectUMFPACK  solver;
         solver.initialize(system_matrix);
         solver.vmult (solution, system_rhs);         
         break;
      }
   }
   
   hanging_node_constraints.distribute (solution);
}



template <int dim>
void OFProblem<dim>::refine_grid ()
{
   switch (refinement_mode) 
   {
      case global_refinement:
      {
         triangulation.refine_global (1);
         break;
      }
         
      case adaptive_refinement:
      {
         Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
         
         typename FunctionMap<dim>::type neumann_boundary;
         KellyErrorEstimator<dim>::estimate (dof_handler,
                                             QGauss<dim-1>(3),
                                             neumann_boundary,
                                             solution,
                                             estimated_error_per_cell);
         
         GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                          estimated_error_per_cell,
                                                          0.3, 0.03);
         
         triangulation.execute_coarsening_and_refinement ();
         
         break;
      }
         
      default:
      {
         Assert (false, ExcNotImplemented());
      }
   }
}



template <int dim>
double OFProblem<dim>::output_results (const unsigned int cycle,double k) const
{
   
   // Project image onto fespace
   FE_Q<dim> feimage (fe.degree);
   DoFHandler<dim> dof_image(triangulation);
   dof_image.distribute_dofs(feimage);
   Vector<double> im(dof_image.n_dofs());
   ConstraintMatrix     hnc;
   hnc.clear ();
   DoFTools::make_hanging_node_constraints (dof_image,
                                            hnc);
   hnc.close ();
   VectorTools::project(dof_image, 
                        hnc, 
                        QGauss<dim>(feimage.degree+1), 
                        Image<dim>(), 
                        im);
   
   // filename for saving image
   std::string filename = "image-";
   filename += ('0' + k);
   filename += ('0' + cycle);
   
   Assert (cycle < 10, ExcInternalError());
   filename += ".vtk";
   std::ofstream image_output (filename.c_str());
   
   DataOut<dim> data_out;

   // save image to file
   data_out.attach_dof_handler (dof_image);
   data_out.add_data_vector (im, "image");
   data_out.build_patches ();
   data_out.write_vtk (image_output);

   data_out.clear ();
   
   // Save solution to file
   filename = "solution-";
   filename += ('0' + k);
   filename += ('0' + cycle);
  
   Assert (cycle < 10, ExcInternalError());
   
   filename += ".vtk";
   std::ofstream output (filename.c_str());
   
   std::vector<DataComponentInterpretation::DataComponentInterpretation> datatype (2);
   datatype[0] = DataComponentInterpretation::component_is_part_of_vector;
   datatype[1] = DataComponentInterpretation::component_is_part_of_vector;

   data_out.attach_dof_handler (dof_handler);
   data_out.add_data_vector (solution, 
                             "velocity", 
                             DataOut_DoFData<DoFHandler<dim>,dim,dim>::type_automatic,
                             datatype);
   
   data_out.build_patches ();
   data_out.write_vtk (output);
   ExactSolution<dim> exact_solution;
   Vector<float> difference_per_cell1 (triangulation.n_active_cells());
   VectorTools::integrate_difference (dof_handler,
				      solution,
				      exact_solution,
				      difference_per_cell1,
				      QGauss<dim>(2),
				      VectorTools::L2_norm);
   double L2_error = difference_per_cell1.l2_norm();
   return L2_error;
   
  
}
template<int dim>
double OFProblem<dim>::advec_err()
{
   QGauss<dim>  quadrature_formula(fe.degree + 1);
   
   FEValues<dim> fe_values (fe, quadrature_formula,
                            update_values   | update_gradients |
                            update_quadrature_points | update_JxW_values);
   
   const unsigned int   dofs_per_cell = fe.dofs_per_cell;
   const unsigned int   n_q_points    = quadrature_formula.size();
   
   
   double local_error;
   
   std::vector<unsigned int> local_dof_indices (dofs_per_cell);
   
   Image<dim> image;
   image.set_time(time);
   
   std::vector<Tensor<1,dim> > image_grad_list (n_q_points, Tensor<1,dim>());
   
   Image_gradt<dim> imaget;
   imaget.set_time(time);
   
   RightHandSide<dim>      right_hand_side;
   right_hand_side.set_time(time);
   std::vector<Vector<double> > rhs_values (n_q_points,
                                            Vector<double>(dim));
   
    
   typename DoFHandler<dim>::active_cell_iterator 

   cell = dof_handler.begin_active(),
   endc = dof_handler.end();
   
   local_error = 0.0;
   double adv;
   for (; cell!=endc; ++cell)
   {
     
      fe_values.reinit (cell);
      
      right_hand_side.vector_value_list (fe_values.get_quadrature_points(),
                                         rhs_values);
      
      image.gradient_list (fe_values.get_quadrature_points(), image_grad_list);
      
    
         const unsigned int
         component_0 = fe.system_to_component_index(0).first;
         
         const unsigned int
         component_1 = fe.system_to_component_index(1).first;
                    
            for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
            {
               adv = (image_grad_list[q_point][component_0] *
                solution(cell->vertex_dof_index(q_point,0)))+
                (image_grad_list[q_point][component_1] *
                solution(cell->vertex_dof_index(q_point,1)))+imaget.value(fe_values.quadrature_point (q_point));
               local_error
               +=
               std::pow(adv,2.0)
                *
                fe_values.JxW(q_point);
            }
        
     }
    return local_error;
     
   
}

template <int dim>
double OFProblem<dim>::run (double k)
{  
   double error;
   for (unsigned int cycle=0; cycle<1; ++cycle)
   {
      std::cout << "Cycle " << cycle << ':' << std::endl;
      
      if (cycle == 0)
      {
         GridGenerator::hyper_cube (triangulation, 0, 1);
         triangulation.refine_global (3);
      }
      else
         refine_grid ();
      
      std::cout << "   Number of active cells:       "
		          << triangulation.n_active_cells()
		          << std::endl;
      
      setup_system ();
      
      std::cout << "   Number of degrees of freedom: "
		          << dof_handler.n_dofs()
                << std::endl;

      assemble_system ();
      
      solve ();
      
       error = output_results (cycle,k);
            
   }
   return error;
}

template class OFProblem<2>;
