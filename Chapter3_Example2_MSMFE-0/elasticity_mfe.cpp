// ---------------------------------------------------------------------
// This code is part of a program that implements the MSMFE methods for Elasticity on various grids:
// - 3D: cuboid and h^2-parallelepiped grids
// - 2D: square, h^2-parallelogram, and distorted quadrilateral grids
//
// Authors:
// Ilona Ambartsumyan, Eldar Khattatov (2016–2017)
// Ibrahim Yazici (2023–2025)
// ---------------------------------------------------------------------

#include <deal.II/base/work_stream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_bdm.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_rt_bubbles.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/base/work_stream.h>

#include <fstream>
#include <cstdio>
#include <iostream>
#include <math.h>

#include "../inc/elasticity_mfe.h"
#include "../inc/elasticity_data.h"
#include "../inc/utilities.h"

#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_dgp_nonparametric.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/base/parameter_handler.h>

namespace elasticity
{
  using namespace dealii;
  using namespace utilities;
  using namespace peflow;
  using namespace std;



  // MixedElasticityProblem: class constructor
  template <int dim>
  MixedElasticityProblem<dim>::MixedElasticityProblem (const unsigned int deg, ParameterHandler &param)
          :
          ElasticityProblem<dim>(deg, param,
                              // FESystem<dim>(FE_RaviartThomas<dim>(deg-1), dim,
                                //             FE_DGQ<dim>(deg-1), dim,
                                  //           FE_DGPNonparametric<dim>(deg-1), static_cast<int>(0.5*dim*(dim-1))))
                                 
                                 FESystem<dim>(FE_RT_Bubbles<dim>(deg), dim,
                                                 FE_DGQ<dim>(deg-1), dim,
                                                 FE_DGQ<dim>(deg-1), static_cast<int>(0.5*dim*(dim-1))))
                                 
                                             //  FE_Q<dim>(deg), static_cast<int>(0.5*dim*(dim-1))))
  {}



  // MixedElasticityProblem: make grid and DoFs
  template <int dim>
  void MixedElasticityProblem<dim>::make_grid_and_dofs()
  {
    TimerOutput::Scope t(computing_timer, "Make grid and DOFs");

    const unsigned int rotation_dim = static_cast<int>(0.5*dim*(dim-1));
    system_matrix.clear();

    dof_handler.distribute_dofs(fe);

    DoFRenumbering::component_wise (dof_handler);

    std::vector<types::global_dof_index> dofs_per_component (dim*dim + dim + rotation_dim);
    dofs_per_component = DoFTools::count_dofs_per_fe_component (dof_handler);
    unsigned int n_s=0, n_u=0, n_p=0;

    for (unsigned int i=0; i<dim; ++i)
    {
      n_s += dofs_per_component[i*dim];
      n_u += dofs_per_component[dim*dim + i];
      // Rotation is scalar in 2d and vector in 3d, so this:
      if (dim == 2)
        n_p = dofs_per_component[dim*dim + dim];
      else if (dim == 3)
        n_p += dofs_per_component[dim*dim + dim + i];
    }

    const unsigned int n_couplings = dof_handler.max_couplings_between_dofs();

    sparsity_pattern.reinit(3,3);
    sparsity_pattern.block(0,0).reinit (n_s, n_s, n_couplings);
    sparsity_pattern.block(1,0).reinit (n_u, n_s, n_couplings);
    sparsity_pattern.block(2,0).reinit (n_p, n_s, n_couplings);

    sparsity_pattern.block(0,1).reinit (n_s, n_u, n_couplings);
    sparsity_pattern.block(1,1).reinit (n_u, n_u, n_couplings);
    sparsity_pattern.block(2,1).reinit (n_p, n_u, n_couplings);

    sparsity_pattern.block(0,2).reinit (n_s, n_p, n_couplings);
    sparsity_pattern.block(1,2).reinit (n_u, n_p, n_couplings);
    sparsity_pattern.block(2,2).reinit (n_p, n_p, n_couplings);
    sparsity_pattern.collect_sizes();

    DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern);
    sparsity_pattern.compress();

    system_matrix.reinit(sparsity_pattern);

    solution.reinit(3);
    solution.block(0).reinit(n_s);
    solution.block(1).reinit(n_u);
    solution.block(2).reinit(n_p);
    solution.collect_sizes();

    system_rhs.reinit(3);
    system_rhs.block(0).reinit(n_s);
    system_rhs.block(1).reinit(n_u);
    system_rhs.block(2).reinit(n_p);
    system_rhs.collect_sizes();
  }

  // Scratch data for multithreading
  template <int dim>
  MixedElasticityProblem<dim>::CellAssemblyScratchData::
  CellAssemblyScratchData (const FiniteElement<dim> &fe,
                           const Quadrature<dim>    &quadrature,
                           const Quadrature<dim-1>  &face_quadrature,
                           const LameCoefficients<dim> &lame_data,
                           Functions::ParsedFunction<dim> *bc,
                           Functions::ParsedFunction<dim> *rhs)
          :
          fe_values (fe,
                     quadrature,
                     update_values   | update_gradients | update_jacobians | update_inverse_jacobians |
                     update_quadrature_points | update_JxW_values),
          fe_face_values (fe,
                          face_quadrature,
                          update_values     | update_quadrature_points   | update_jacobians | update_inverse_jacobians |
                          update_JxW_values | update_normal_vectors),
          lame(lame_data),
          bc(bc),
          rhs(rhs)
  {}

  template <int dim>
  MixedElasticityProblem<dim>::CellAssemblyScratchData::
  CellAssemblyScratchData (const CellAssemblyScratchData &scratch_data)
          :
          fe_values (scratch_data.fe_values.get_fe(),
                     scratch_data.fe_values.get_quadrature(),
                     update_values   | update_gradients | update_jacobians | update_inverse_jacobians |
                     update_quadrature_points | update_JxW_values),
          fe_face_values (scratch_data.fe_face_values.get_fe(),
                          scratch_data.fe_face_values.get_quadrature(),
                          update_values     | update_quadrature_points   | update_jacobians | update_inverse_jacobians |
                          update_JxW_values | update_normal_vectors),
          lame(scratch_data.lame),
          bc(scratch_data.bc),
          rhs(scratch_data.rhs)
  {}


  // Copy local contributions to global system
  template <int dim>
  void MixedElasticityProblem<dim>::copy_local_to_global (const CellAssemblyCopyData &copy_data)
  {
    for (unsigned int i=0; i<copy_data.local_dof_indices.size(); ++i)
    {
      for (unsigned int j=0; j<copy_data.local_dof_indices.size(); ++j)
        system_matrix.add (copy_data.local_dof_indices[i],
                           copy_data.local_dof_indices[j],
                           copy_data.cell_matrix(i,j));
      system_rhs(copy_data.local_dof_indices[i]) += copy_data.cell_rhs(i);
    }
  }



  // Function to assemble on a cell
  template <int dim>
  void MixedElasticityProblem<dim>::assemble_system_cell (const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                          CellAssemblyScratchData                                   &scratch_data,
                                                          CellAssemblyCopyData                                      &copy_data)
  {
    const unsigned int rotation_dim    = static_cast<int>(0.5*dim*(dim-1));
    const unsigned int dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int n_q_points      = scratch_data.fe_values.get_quadrature().size();
    const unsigned int n_face_q_points = scratch_data.fe_face_values.get_quadrature().size();

    copy_data.cell_matrix.reinit (dofs_per_cell, dofs_per_cell);
    copy_data.cell_rhs.reinit (dofs_per_cell);
    copy_data.local_dof_indices.resize(dofs_per_cell);

    scratch_data.fe_values.reinit (cell);

    // Stress and rotation DoFs vectors
    std::vector<FEValuesExtractors::Vector> stresses(dim, FEValuesExtractors::Vector());
    std::vector<FEValuesExtractors::Scalar> rotations(rotation_dim, FEValuesExtractors::Scalar());

    // Displacement DoFs
    const FEValuesExtractors::Vector displacement (dim*dim);

    for (unsigned int d=0; d<dim; ++d)
    {
      const FEValuesExtractors::Vector tmp_stress(d*dim);
      stresses[d].first_vector_component = tmp_stress.first_vector_component;
      if (dim == 2 && d == 0)
      {
        const FEValuesExtractors::Scalar tmp_rotation(dim*dim + dim);
        rotations[d].component = tmp_rotation.component;
      } else if (dim == 3) {
        const FEValuesExtractors::Scalar tmp_rotation(dim*dim + dim + d);
        rotations[d].component = tmp_rotation.component;
      }
    }

    // Stress, divergence and rotation
    std::vector<std::vector<Tensor<1,dim>>> phi_i_s(dofs_per_cell, std::vector<Tensor<1,dim>> (dim));
    std::vector<Tensor<1,dim>> div_phi_i_s(dofs_per_cell);
    std::vector<Tensor<1,dim>> phi_i_u(dofs_per_cell);
    std::vector<Tensor<1,rotation_dim>> phi_i_p(dofs_per_cell);

    for (unsigned int q=0; q<n_q_points; ++q)
    {
        
      for (unsigned int k=0; k<dofs_per_cell; ++k)
      {
          
        // Evaluate test functions
        for (unsigned int s_i=0; s_i<dim; ++s_i)
        {
          phi_i_s[k][s_i] = scratch_data.fe_values[stresses[s_i]].value (k, q);
          div_phi_i_s[k][s_i] = scratch_data.fe_values[stresses[s_i]].divergence (k, q);
        }
        phi_i_u[k] = scratch_data.fe_values[displacement].value (k, q);
        for (unsigned int r_i=0; r_i<rotation_dim; ++r_i)
          phi_i_p[k][r_i] = scratch_data.fe_values[rotations[r_i]].value (k, q);
      }
        
        // Only needed for dsc params
      const Point<dim> center = cell->center();
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
        Point<dim> point = scratch_data.fe_values.get_quadrature_points()[q];
        const double mu = scratch_data.lame.mu_value(center);
        const double lambda = scratch_data.lame.lambda_value(center);
        //Tensor<2,dim> asigma = compliance_tensor_stress<dim>(phi_i_s[i], mu, lambda);
        for (unsigned int j=0; j<dofs_per_cell; ++j)
        {
          Tensor<2,dim> asigma = compliance_tensor_stress<dim>(phi_i_s[j], mu, lambda);
          Tensor<2,dim> sigma = make_tensor(phi_i_s[i]);
            
            double bl_sum = 0.0;
            
            bl_sum = (            scalar_product(asigma, sigma)
                                           + scalar_product(phi_i_u[i], div_phi_i_s[j])
                                           + scalar_product(phi_i_u[j], div_phi_i_s[i])
                                           + scalar_product(phi_i_p[i], make_asymmetry_tensor(phi_i_s[j]))
                                           + scalar_product(phi_i_p[j], make_asymmetry_tensor(phi_i_s[i]))) * scratch_data.fe_values.JxW(q);
             
            // Section implementing non-symmetric quadrature rules starts here
            if (dim ==2){
                
                double DFrc[2][2]={};
                DFrc[0][0]=(scratch_data.fe_values.jacobian(0)[0][0]+scratch_data.fe_values.jacobian(1)[0][0]
                +scratch_data.fe_values.jacobian(2)[0][0]+scratch_data.fe_values.jacobian(3)[0][0])/(4.0);
                DFrc[0][1]=(scratch_data.fe_values.jacobian(0)[0][1]+scratch_data.fe_values.jacobian(1)[0][1]
                +scratch_data.fe_values.jacobian(2)[0][1]+scratch_data.fe_values.jacobian(3)[0][1])/(4.0);
                DFrc[1][0]=(scratch_data.fe_values.jacobian(0)[1][0]+scratch_data.fe_values.jacobian(1)[1][0]
                +scratch_data.fe_values.jacobian(2)[1][0]+scratch_data.fe_values.jacobian(3)[1][0])/(4.0);
                DFrc[1][1]=(scratch_data.fe_values.jacobian(0)[1][1]+scratch_data.fe_values.jacobian(1)[1][1]
                +scratch_data.fe_values.jacobian(2)[1][1]+scratch_data.fe_values.jacobian(3)[1][1])/(4.0);
                
                double temp[2][2]={};
                temp[0][0]=sigma[0][0]*scratch_data.fe_values.inverse_jacobian(q)[0][0]
                + sigma[0][1]*scratch_data.fe_values.inverse_jacobian(q)[0][1];
                temp[0][1]=sigma[0][0]*scratch_data.fe_values.inverse_jacobian(q)[1][0]
                + sigma[0][1]*scratch_data.fe_values.inverse_jacobian(q)[1][1];
                temp[1][0]=sigma[1][0]*scratch_data.fe_values.inverse_jacobian(q)[0][0]
                + sigma[1][1]*scratch_data.fe_values.inverse_jacobian(q)[0][1];
                temp[1][1]=sigma[1][0]*scratch_data.fe_values.inverse_jacobian(q)[1][0]
                + sigma[1][1]*scratch_data.fe_values.inverse_jacobian(q)[1][1];
                
                double temp2[2][2]={};
                temp2[0][0]=temp[0][0]*DFrc[0][0]
                + temp[0][1]*DFrc[0][1];
                temp2[0][1]=temp[0][0]*DFrc[1][0]
                + temp[0][1]*DFrc[1][1];
                temp2[1][0]=temp[1][0]*DFrc[0][0]
                + temp[1][1]*DFrc[0][1];
                temp2[1][1]=temp[1][0]*DFrc[1][0]
                + temp[1][1]*DFrc[1][1];
                
                double sigma_new[2][2]={};
                sigma_new[0][0]=temp2[0][0];
                sigma_new[0][1]=temp2[0][1];
                sigma_new[1][0]=temp2[1][0];
                sigma_new[1][1]=temp2[1][1];
                
                Tensor<2,dim> rotthrd = make_tensor(phi_i_s[j]);
                
                temp[0][0]=rotthrd[0][0]*scratch_data.fe_values.inverse_jacobian(q)[0][0]
                + rotthrd[0][1]*scratch_data.fe_values.inverse_jacobian(q)[0][1];
                temp[0][1]=rotthrd[0][0]*scratch_data.fe_values.inverse_jacobian(q)[1][0]
                + rotthrd[0][1]*scratch_data.fe_values.inverse_jacobian(q)[1][1];
                temp[1][0]=rotthrd[1][0]*scratch_data.fe_values.inverse_jacobian(q)[0][0]
                + rotthrd[1][1]*scratch_data.fe_values.inverse_jacobian(q)[0][1];
                temp[1][1]=rotthrd[1][0]*scratch_data.fe_values.inverse_jacobian(q)[1][0]
                + rotthrd[1][1]*scratch_data.fe_values.inverse_jacobian(q)[1][1];
                
                temp2[0][0]=temp[0][0]*DFrc[0][0]
                + temp[0][1]*DFrc[0][1];
                temp2[0][1]=temp[0][0]*DFrc[1][0]
                + temp[0][1]*DFrc[1][1];
                temp2[1][0]=temp[1][0]*DFrc[0][0]
                + temp[1][1]*DFrc[0][1];
                temp2[1][1]=temp[1][0]*DFrc[1][0]
                + temp[1][1]*DFrc[1][1];
                
                double rotthrd_new[2][2]={};
                rotthrd_new[0][0]=temp2[0][0];
                rotthrd_new[0][1]=temp2[0][1];
                rotthrd_new[1][0]=temp2[1][0];
                rotthrd_new[1][1]=temp2[1][1];
                
                Tensor<2,dim> rotfrst = make_tensor(phi_i_s[i]);
                
                temp[0][0]=rotfrst[0][0]*scratch_data.fe_values.inverse_jacobian(q)[0][0]
                + rotfrst[0][1]*scratch_data.fe_values.inverse_jacobian(q)[0][1];
                temp[0][1]=rotfrst[0][0]*scratch_data.fe_values.inverse_jacobian(q)[1][0]
                + rotfrst[0][1]*scratch_data.fe_values.inverse_jacobian(q)[1][1];
                temp[1][0]=rotfrst[1][0]*scratch_data.fe_values.inverse_jacobian(q)[0][0]
                + rotfrst[1][1]*scratch_data.fe_values.inverse_jacobian(q)[0][1];
                temp[1][1]=rotfrst[1][0]*scratch_data.fe_values.inverse_jacobian(q)[1][0]
                + rotfrst[1][1]*scratch_data.fe_values.inverse_jacobian(q)[1][1];
                
                temp2[0][0]=temp[0][0]*DFrc[0][0]
                + temp[0][1]*DFrc[0][1];
                temp2[0][1]=temp[0][0]*DFrc[1][0]
                + temp[0][1]*DFrc[1][1];
                temp2[1][0]=temp[1][0]*DFrc[0][0]
                + temp[1][1]*DFrc[0][1];
                temp2[1][1]=temp[1][0]*DFrc[1][0]
                + temp[1][1]*DFrc[1][1];
                
                double rotfrst_new[2][2]={};
                rotfrst_new[0][0]=temp2[0][0];
                rotfrst_new[0][1]=temp2[0][1];
                rotfrst_new[1][0]=temp2[1][0];
                rotfrst_new[1][1]=temp2[1][1];
                
                double phi_i_p_i_term=phi_i_p[i][0];
                double phi_i_p_j_term=phi_i_p[j][0];
                
                bl_sum = (asigma[0][0]*sigma_new[0][0]+asigma[0][1]*sigma_new[0][1]+asigma[1][0]*sigma_new[1][0]+asigma[1][1]*sigma_new[1][1]) * scratch_data.fe_values.JxW(q);
                bl_sum = bl_sum + (phi_i_p_i_term*rotthrd[0][1] - phi_i_p_i_term*rotthrd[1][0])* scratch_data.fe_values.JxW(q);
                bl_sum = bl_sum + (phi_i_p_j_term*rotfrst_new[0][1] - phi_i_p_j_term*rotfrst_new[1][0])* scratch_data.fe_values.JxW(q);
                bl_sum = bl_sum + (scalar_product(phi_i_u[i], div_phi_i_s[j]) + scalar_product(phi_i_u[j], div_phi_i_s[i])) * scratch_data.fe_values.JxW(q);
            }
            // Section implementing non-symmetric quadrature rules ends here
            
            copy_data.cell_matrix(i,j) += bl_sum;
        }

        for (unsigned d_i=0; d_i<dim; ++d_i)
          copy_data.cell_rhs(i) += -(phi_i_u[i][d_i] * scratch_data.rhs->value(scratch_data.fe_values.get_quadrature_points()[q], d_i)) * scratch_data.fe_values.JxW(q);
      }
    }

   // for (unsigned int i=0; i<dofs_per_cell; ++i)
   //   for (unsigned int j=i+1; j<dofs_per_cell; ++j)
   //     copy_data.cell_matrix(j,i) = copy_data.cell_matrix(i,j);

    for (unsigned int face_no=0;
         face_no<GeometryInfo<dim>::faces_per_cell;
         ++face_no)
      if (cell->at_boundary(face_no)) // && (cell->face(face_no)->boundary_id() == 1)
      {
        scratch_data.fe_face_values.reinit (cell, face_no);

        for (unsigned int q=0; q<n_face_q_points; ++q)
          for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            Tensor<2,dim> sigma;
            for (unsigned int d_i=0; d_i<dim; ++d_i)
              sigma[d_i] = scratch_data.fe_face_values[stresses[d_i]].value (i, q);

            Tensor<1,dim> sigma_n = sigma * scratch_data.fe_face_values.normal_vector(q);
            for (unsigned int d_i=0; d_i<dim; ++d_i)
              copy_data.cell_rhs(i) += ((sigma_n[d_i]*scratch_data.bc->value(scratch_data.fe_face_values.get_quadrature_points()[q],d_i))
                                       *scratch_data.fe_face_values.JxW(q));

          }
      }
    cell->get_dof_indices (copy_data.local_dof_indices);
  }

  template <int dim>
  void MixedElasticityProblem<dim>::assemble_system ()
  {
    Functions::ParsedFunction<dim> *mu                  = new Functions::ParsedFunction<dim>(1);
    Functions::ParsedFunction<dim> *lambda              = new Functions::ParsedFunction<dim>(1);
    Functions::ParsedFunction<dim> *bc       = new Functions::ParsedFunction<dim>(dim);
    Functions::ParsedFunction<dim> *rhs      = new Functions::ParsedFunction<dim>(dim);

    prm.enter_subsection(std::string("lambda ") + Utilities::int_to_string(dim)+std::string("D"));
    lambda->parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection(std::string("mu ") + Utilities::int_to_string(dim)+std::string("D"));
    mu->parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection(std::string("BC ")+ Utilities::int_to_string(dim)+std::string("D"));
    bc->parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection(std::string("RHS ") + Utilities::int_to_string(dim)+std::string("D"));
    rhs->parse_parameters(prm);
    prm.leave_subsection();

    LameCoefficients<dim> lame(prm,mu, lambda);

    TimerOutput::Scope t(computing_timer, "Assemble system");
      
    
    QGaussLobatto<dim> quad(degree+1);
    QGauss<dim-1> face_quad(degree);
    //QGauss<dim-1> face_quad(2*(degree+1)+1);

    //QGauss<dim> quad(2*(degree+1)+1);
    //QGauss<dim-1> face_quad(2*(degree+1)+1);
      
    WorkStream::run(dof_handler.begin_active(),
                    dof_handler.end(),
                    *this,
                    &MixedElasticityProblem::assemble_system_cell,
                    &MixedElasticityProblem::copy_local_to_global,
                    CellAssemblyScratchData(fe,quad,face_quad, lame, bc, rhs),
                    CellAssemblyCopyData());

    delete mu;
    delete lambda;
    delete bc;
    delete rhs;
  }

  // MixedElasticityProblem: Solve
  template <int dim>
  void MixedElasticityProblem<dim>::solve ()
  {
    TimerOutput::Scope t(computing_timer, "Solve (Direct UMFPACK)");

    SparseDirectUMFPACK  A_direct;
    A_direct.initialize(system_matrix);
    A_direct.vmult (solution, system_rhs);
  }

  // MixedElasticityProblem: run
  template <int dim>
  void MixedElasticityProblem<dim>::run(const unsigned int refine, const unsigned int grid)
  {
    dof_handler.clear();
    triangulation.clear();
    convergence_table.clear();

    for (unsigned int cycle=0; cycle<refine; ++cycle)
    {

        if(cycle == 0 || grid >= 3){
                   if(grid == 1){
                       GridIn<dim> grid_in;
                       grid_in.attach_triangulation (triangulation);
                       std::string mesh_filename ("mesh"+std::to_string(dim)+"d.msh");
                       std::ifstream input_file(mesh_filename);

                       Assert(input_file.is_open(), ExcFileNotOpen(mesh_filename.c_str()));
                       Assert(triangulation.dimension == dim, ExcDimensionMismatch(triangulation.dimension, dim));

                       grid_in.read_msh (input_file);
                     } else if (grid == 0) {
                       GridGenerator::hyper_cube (triangulation, 0, 1);
                       triangulation.refine_global(1);
                     } else if (grid == 2) {
                       GridGenerator::subdivided_hyper_cube(triangulation, 3, 0.0, 1.0);
                     } else if (grid == 3) {
                         triangulation.clear();
                         GridGenerator::hyper_cube (triangulation, 0, 1);
                       triangulation.refine_global(cycle+1);
                       GridTools::transform(&grid_transform_h2<dim>, triangulation);
                     } else if (grid > 3) {
                       // h^k uniform perturbation
                         triangulation.clear();
                         GridGenerator::hyper_cube (triangulation, 0, 1);
                       const unsigned int refinements = cycle+2;
                       triangulation.refine_global(refinements);

                       double reg = 0.0;
                       if (grid == 4)
                         reg = 2.0;
                       else if (grid == 5)
                         reg = 1.5;
                       else if (grid == 6)
                         reg = 1.0;

                       auto rand_engine = std::mt19937(77);
                       std::mt19937* const rand_engine_ptr = &rand_engine;
                       auto func = std::bind(grid_transform_unit_hk<dim>,
                                             std::placeholders::_1,
                                             rand_engine_ptr,
                                             reg,
                                             1.0/std::pow(2,refinements));
                       GridTools::transform(func, triangulation);
                     }
        
        typename Triangulation<dim>::cell_iterator
                cell = triangulation.begin (),
                endc = triangulation.end();
        for (; cell!=endc; ++cell)
          for (unsigned int face_number=0;
               face_number<GeometryInfo<dim>::faces_per_cell;
               ++face_number)
            if ((std::fabs(cell->face(face_number)->center()(0) - (1)) < 1e-12)
                ||
                (std::fabs(cell->face(face_number)->center()(1) - (1)) < 1e-12))
              cell->face(face_number)->set_boundary_id (1);
      } else {
        triangulation.refine_global(1);
      }

      make_grid_and_dofs();
      assemble_system ();
      solve ();
      compute_errors(cycle);
      output_results (cycle, refine);
      computing_timer.print_summary();
      computing_timer.reset();
    }
  }

  template class MixedElasticityProblem<2>;
  template class MixedElasticityProblem<3>;
}

