#ifndef __IMAGE_H__
#define __IMAGE_H__

#include <base/function.h>
#include <base/function_time.h>

using namespace dealii;

template <int dim>
class RightHandSide :  public Function<dim>
{
public:
   RightHandSide() : Function<dim>(dim) {};
   
   
   virtual void vector_value (const Point<dim> &p,
                              Vector<double>   &values) const;
    virtual void vector_value_list (const std::vector<Point<dim> > &points,
                                   std::vector<Vector<double> >   &value_list) const;
   
};


template <int dim>
class Image :  public Function<dim>
{
public:
   Image() : Function<dim>() {}
   virtual double value(const Point<dim> &p, 
                        const unsigned int component = 0) const;
   virtual Tensor<1,dim> gradient(const Point<dim> &p, 
                                  const unsigned int component = 0) const;
   virtual void gradient_list(const std::vector<Point<dim> > &p, 
                              std::vector<Tensor<1,dim> > &grad_values, 
                              const unsigned int component = 0) const;
   
};

template<int dim>
class Image_gradt :  public Function<dim>
{
public:
   Image_gradt() : Function<dim>() {}
   virtual double value(const Point<dim> &p, 
                        const unsigned int component = 0) const;
};

template <int dim>
 class ExactSolution : public Function<dim> 
 {
   public:
     ExactSolution () : Function<dim>(dim) {}
     
     virtual void vector_value (const Point<dim> &p, 
                                Vector<double>   &value) const;
 };


#endif
