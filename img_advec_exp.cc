/* To calculate the velocity in optical flow problem */


#include <base/function.h>
#include <base/function_time.h>

#include <numerics/vectors.h>



#include "image.h"

using namespace dealii;


//------------------------------------------------------------------------------

template<int dim>
double Image<dim>::value(const Point<dim> &p, 
                         const unsigned int) const
{
   double x = p[0];
   double y = p[1];
   double t = this->get_time();
   double arg = -50*(std::pow((x-t-0.5),2.0)+std::pow((y-t-0.5),2.0));
   
   double return_value = std::exp(arg);
   
   return return_value;
}

template<int dim>
Tensor<1,dim> Image<dim>::gradient(const Point<dim> &p, 
                                   const unsigned int)const
{
   Assert (dim == 2,ExcIndexRange (dim, 2, 3));
   
   double x = p[0];
   double y = p[1];
   double t = this->get_time();
   
   Tensor<1,dim> return_value;
   
   double arg = -50*(std::pow((x-t-0.5),2.0)+std::pow((y-t-0.5),2.0));
   return_value[0] = -std::exp(arg)*2.0*50.0*(x-t-0.5);
   return_value[1] = -std::exp(arg)*2.0*50.0*(y-t-0.5);
      
   return return_value;
}

template<int dim>
void Image<dim>::gradient_list(const std::vector<Point<dim> > &p, 
                               std::vector<Tensor<1,dim> > &grad_values, 
                               const unsigned int ) const
{
   Assert (grad_values.size() == p.size(),
           ExcDimensionMismatch (grad_values.size(), p.size()));
   
   const unsigned int np = p.size();
   
   for(unsigned int i=0; i<np; ++i)
   {
      grad_values[i] = this->gradient(p[i]);
   }
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
 template <int dim>
 void ExactSolution<dim>::vector_value (const Point<dim> &p,
                                   Vector<double>   &values) const 
 {
   Assert (values.size() == dim,
           ExcDimensionMismatch (values.size(), dim));
 
   
   values(0) = std::cos(M_PI*p(0));
   values(1) = std::sin(3.0*M_PI*p(1)/2.0);
   
 }

template<int dim>
double Image_gradt<dim>::value(const Point<dim> &p, 
                               const unsigned int) const
{
   Assert (dim == 2,ExcIndexRange (dim, 2, 3));
   
      
   Image<dim> image;
   ExactSolution<dim> exact;
   
   Vector<double> exact_sol(dim);
   
   exact.vector_value(p,exact_sol);
   
   double return_value = -image.gradient(p)[0]*exact_sol(0)-image.gradient(p)[1]*exact_sol(1);
   
   return return_value;
}				 


template <int dim>
void RightHandSide<dim>::vector_value (const Point<dim> &p,
                                       Vector<double>   &values) const
{
   Assert (values.size() == dim,
           ExcDimensionMismatch (values.size(), dim));
   Assert (dim == 2, ExcNotImplemented());
   
   double t = this->get_time();
      
   Image<dim> image;
   image.set_time(t);
   
   Image_gradt<dim> imaget;
   imaget.set_time(t);
   
   values(0) = -imaget.value(p)*image.gradient(p)[0];
   values(1) = -imaget.value(p)*image.gradient(p)[1];
   
}

template <int dim>
void RightHandSide<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                            std::vector<Vector<double> >   &value_list) const
{
   Assert (value_list.size() == points.size(),
           ExcDimensionMismatch (value_list.size(), points.size()));
   
   const unsigned int n_points = points.size();
   
   
   for (unsigned int p=0; p<n_points; ++p)
      this->vector_value (points[p], value_list[p]);
}

template class Image<2>;
template class Image_gradt<2>;
template class ExactSolution<2>;
template class RightHandSide<2>;;

