// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2020 Eldar Khattatov
//
// This file is part of peFLOW.
//
// This code is part of a program that implements the MFMFE methods for Darcy flow on various grids:
// - 3D: cuboid and h^2-parallelepiped grids
// - 2D: square, and h^2-parallelogram grids
//
// ---------------------------------------------------------------------

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_rt_bubbles.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/base/work_stream.h>

#include <fstream>

#include "../inc/problem.h"
#include "../inc/darcy_data.h"
#include "../inc/darcy_mfmfe.h"
#include "../inc/utilities.h"
#include <deal.II/base/parameter_handler.h>

// TODO: Refactor accoding to hMFMFE tutorial

namespace darcy
{
  using namespace dealii;
  using namespace utilities;
  using namespace peflow;

  // MultipointMixedDarcyProblem: class constructor
  template <int dim>
  MultipointMixedDarcyProblem<dim>::MultipointMixedDarcyProblem (const unsigned int degree, ParameterHandler &param)
          :
          DarcyProblem<dim>(degree, param,
                            FESystem<dim>(FE_RT_Bubbles<dim>(degree), 1, FE_DGQ<dim>(degree-1), 1))
  {}


  template <int dim>
  void MultipointMixedDarcyProblem<dim>::reset_data_structures ()
  {
    pressure_indices.clear();
    velocity_indices.clear();
    velocity_rhs.clear();
    A_inverse.clear();
    pressure_matrix.clear();
    vertex_matrix.clear();
    vertex_rhs.clear();
  }


// Scratch data for multithreading
  template <int dim>
  MultipointMixedDarcyProblem<dim>::VertexAssemblyScratchData::
  VertexAssemblyScratchData (const FiniteElement<dim> &fe,
                             const Triangulation<dim> &tria,
                             const Quadrature<dim> &quad,
                             const Quadrature<dim-1> &f_quad,
                             const KInverse<dim> &k_data,
                             Functions::ParsedFunction<dim> *bc,
                             Functions::ParsedFunction<dim> *rhs)
          :
          fe_values (fe,
                     quad,
                     update_values   | update_gradients | update_jacobians | update_inverse_jacobians |
                     update_quadrature_points | update_JxW_values),
          fe_face_values (fe,
                          f_quad,
                          update_values     | update_quadrature_points |update_jacobians | update_inverse_jacobians |
                          update_JxW_values | update_normal_vectors),
          K_inv(k_data),
          bc(bc),
          rhs(rhs),
          num_cells(tria.n_active_cells())
  {
    n_faces_at_vertex.resize(tria.n_vertices(), 0);
    typename Triangulation<dim>::active_face_iterator face = tria.begin_active_face(), endf = tria.end_face();

    for (; face != endf; ++face)
      for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_face; ++v)
        n_faces_at_vertex[face->vertex_index(v)] += 1;
  }

  template <int dim>
  MultipointMixedDarcyProblem<dim>::VertexAssemblyScratchData::
  VertexAssemblyScratchData (const VertexAssemblyScratchData &scratch_data)
          :
          fe_values (scratch_data.fe_values.get_fe(),
                     scratch_data.fe_values.get_quadrature(),
                     update_values   | update_gradients | update_jacobians | update_inverse_jacobians |
                     update_quadrature_points | update_JxW_values),
          fe_face_values (scratch_data.fe_face_values.get_fe(),
                          scratch_data.fe_face_values.get_quadrature(),
                          update_values     | update_quadrature_points | update_jacobians | update_inverse_jacobians |
                          update_JxW_values | update_normal_vectors),
          n_faces_at_vertex(scratch_data.n_faces_at_vertex),
          K_inv(scratch_data.K_inv),
          bc(scratch_data.bc),
          rhs(scratch_data.rhs),
          num_cells(scratch_data.num_cells)
  {}


  template <int dim>
  void MultipointMixedDarcyProblem<dim>::copy_cell_to_vertex (const VertexAssemblyCopyData &copy_data)
  {
    for (const auto &m : copy_data.cell_mat)
    {
      for (const auto &p : m.second)
        vertex_matrix[m.first][p.first] += p.second;

      for (const auto &p : copy_data.cell_vec.at(m.first))
        vertex_rhs[m.first][p.first] += p.second;

      for (const auto &p : copy_data.local_pres_indices.at(m.first))
        pressure_indices[m.first].insert(p);


      for (const auto &p : copy_data.local_vel_indices.at(m.first))
        velocity_indices[m.first].insert(p);
    }
  }

// Function to assemble on a cell
  template <int dim>
  void MultipointMixedDarcyProblem<dim>::assemble_system_cell (const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                               VertexAssemblyScratchData                         &scratch_data,
                                                               VertexAssemblyCopyData                            &copy_data)
  {
    copy_data.cell_mat.clear();
    copy_data.cell_vec.clear();
    copy_data.local_vel_indices.clear();
    copy_data.local_pres_indices.clear();

    const unsigned int dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int n_q_points      = scratch_data.fe_values.get_quadrature().size();
    const unsigned int n_face_q_points = scratch_data.fe_face_values.get_quadrature().size();

    copy_data.local_dof_indices.resize(dofs_per_cell);
    cell->get_dof_indices (copy_data.local_dof_indices);

    scratch_data.fe_values.reinit (cell);

    std::vector<Tensor<2,dim>>             k_inverse_values (n_q_points);

    scratch_data.K_inv.value_list (scratch_data.fe_values.get_quadrature_points(), k_inverse_values);


    // Velocity and pressure DoFs vectors
    const FEValuesExtractors::Vector velocity (0);
    const FEValuesExtractors::Scalar pressure (dim);

    unsigned int n_vel = dim*pow(degree+1,dim);
    std::unordered_map<unsigned int, std::unordered_map<unsigned int, double>> div_map;

    // Assemble the div terms
    for (unsigned int q=0; q<n_q_points; ++q)
    {
      Point<dim> p = scratch_data.fe_values.quadrature_point(q);
      round_point(p);

      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
        const double div_phi_i_u = scratch_data.fe_values[velocity].divergence (i, q);
        const double phi_i_p     = scratch_data.fe_values[pressure].value (i, q);
        for (unsigned int j=n_vel; j<dofs_per_cell; ++j)
        {
          const double div_phi_j_u = scratch_data.fe_values[velocity].divergence (j, q);
          const double phi_j_p     = scratch_data.fe_values[pressure].value (j, q);

          double div_term = (- div_phi_i_u * phi_j_p - phi_i_p * div_phi_j_u) * scratch_data.fe_values.JxW(q);

          if (fabs(div_term) > 1.e-12)
            div_map[i][j] += div_term;
        }

        double source_term = -phi_i_p *scratch_data.rhs->value(scratch_data.fe_values.get_quadrature_points()[q]) *scratch_data.fe_values.JxW(q);

        bool pres_flag = false;
        if (fabs(phi_i_p) > 1.e-12)
          pres_flag = true;

        if (pres_flag || fabs(source_term) > 1.e-12) {
          copy_data.cell_vec[p][copy_data.local_dof_indices[i]] += source_term;
        }
      }
    }

    // Assemble coercive terms and incorporate div terms
    for (unsigned int q=0; q<n_q_points; ++q)
    {
      std::set<types::global_dof_index> vel_indices;
      Point<dim> p = scratch_data.fe_values.quadrature_point(q);
      round_point(p);
        
        
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
         Tensor<1,dim> phi_i_u     = scratch_data.fe_values[velocity].value (i, q);
        for (unsigned int j=i; j<dofs_per_cell; ++j)
        {
          Tensor<1,dim> phi_j_u     = scratch_data.fe_values[velocity].value (j, q);
            
            double mass_term = phi_i_u * k_inverse_values[q] * phi_j_u * scratch_data.fe_values.JxW(q);

          if (fabs(mass_term) > 1.e-12) {
            copy_data.cell_mat[p][std::make_pair(copy_data.local_dof_indices[i], copy_data.local_dof_indices[j])] +=
                    mass_term;
            vel_indices.insert(i);
            copy_data.local_vel_indices[p].insert(copy_data.local_dof_indices[j]);
          }
        }
      }

      for (const auto &i : vel_indices)
        for (const auto &el : div_map[i]) {
          if (fabs(el.second) > 1.e-12) {
            copy_data.cell_mat[p][std::make_pair(copy_data.local_dof_indices[i],
                                                 copy_data.local_dof_indices[el.first])] += el.second;
            copy_data.local_pres_indices[p].insert(copy_data.local_dof_indices[el.first]);
          }
        }
    }

    std::map<types::global_dof_index,double> pres_bc;
    for (unsigned int face_no=0;
         face_no<GeometryInfo<dim>::faces_per_cell;
         ++face_no)
      if ( (cell->at_boundary(face_no)) ) // && (cell->face(face_no)->boundary_id() != 1)
      {
        scratch_data.fe_face_values.reinit (cell, face_no);

        for (unsigned int q=0; q<n_face_q_points; ++q) {
          for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            double tmp = -(scratch_data.fe_face_values[velocity].value(i, q) *
                           scratch_data.fe_face_values.normal_vector(q) *
                           scratch_data.bc->value(scratch_data.fe_face_values.get_quadrature_points()[q]) *
                           scratch_data.fe_face_values.JxW(q));

            if (fabs(tmp) > 1.e-12)
              pres_bc[copy_data.local_dof_indices[i]] += tmp;

          }
        }
      }

    for (const auto &m : copy_data.cell_vec)
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        if (fabs(pres_bc[copy_data.local_dof_indices[i]]) > 1.e-12)
          copy_data.cell_vec[m.first][copy_data.local_dof_indices[i]] += pres_bc[copy_data.local_dof_indices[i]];
  }


  template <int dim>
  void MultipointMixedDarcyProblem<dim>::vertex_assembly()
  {
    Functions::ParsedFunction<dim> *k_inv = new Functions::ParsedFunction<dim>(dim*dim);
    Functions::ParsedFunction<dim> *bc       = new Functions::ParsedFunction<dim>(1);
    Functions::ParsedFunction<dim> *rhs      = new Functions::ParsedFunction<dim>(1);

    prm.enter_subsection(std::string("permeability ") + Utilities::int_to_string(dim)+std::string("D"));
    k_inv->parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection(std::string("BC ") + Utilities::int_to_string(dim)+std::string("D"));
    bc->parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection(std::string("RHS ") + Utilities::int_to_string(dim)+std::string("D"));
    rhs->parse_parameters(prm);
    prm.leave_subsection();

    KInverse<dim> k_inverse(prm,k_inv);

    TimerOutput::Scope t(computing_timer, "Vertex assembly");

    dof_handler.distribute_dofs(fe);
    DoFRenumbering::component_wise (dof_handler);
    std::vector<types::global_dof_index> dofs_per_component (dim+1);
    dofs_per_component = DoFTools::count_dofs_per_fe_component (dof_handler);

    QGaussLobatto<dim> quad(degree+1);
    QGauss<dim-1> face_quad(degree);

    n_v = dofs_per_component[0];
    n_p = dofs_per_component[dim];

    pres_rhs.reinit(n_p);

    WorkStream::run(dof_handler.begin_active(),
                    dof_handler.end(),
                    *this,
                    &MultipointMixedDarcyProblem::assemble_system_cell,
                    &MultipointMixedDarcyProblem::copy_cell_to_vertex,
                    VertexAssemblyScratchData(fe, triangulation, quad, face_quad,
                                              k_inverse, bc, rhs),
                    VertexAssemblyCopyData());

    delete k_inv;
    delete bc;
    delete rhs;
  }

  template <int dim>
  void MultipointMixedDarcyProblem<dim>::make_cell_centered_sp()
  {
    TimerOutput::Scope t(computing_timer, "Make sparsity pattern");
    DynamicSparsityPattern dsp(n_p, n_p);

    std::set<types::global_dof_index>::iterator pi_it, pj_it;
    unsigned int i, j;
    for (const auto &el : vertex_matrix)
      for (pi_it = pressure_indices[el.first].begin(), i = 0;
           pi_it != pressure_indices[el.first].end();
           ++pi_it, ++i) {
        for (pj_it = pi_it, j = 0;
             pj_it != pressure_indices[el.first].end();
             ++pj_it, ++j)
          dsp.add(*pi_it - n_v, *pj_it - n_v);
      }


    dsp.symmetrize();
    cell_centered_sp.copy_from(dsp);
    pres_system_matrix.reinit (cell_centered_sp);
  }


  template <int dim>
  void MultipointMixedDarcyProblem<dim>::vertex_elimination(const typename MapPointMatrix<dim>::iterator &n_it,
                                                            VertexAssemblyScratchData &scratch_data,
                                                            VertexEliminationCopyData &copy_data)
  {
    // Matrix for each component
    unsigned int n_edges = velocity_indices.at((*n_it).first).size();
    unsigned int n_cells = pressure_indices.at((*n_it).first).size();

    FullMatrix<double> velocity_matrix(n_edges,n_edges);
    copy_data.pressure_matrix.reinit(n_edges,n_cells);

    //RHS vector for each component
    copy_data.velocity_rhs.reinit(n_edges);
    Vector<double> pressure_rhs(n_cells);

    std::set<types::global_dof_index>::iterator vi_it, vj_it, p_it;
    unsigned int i, j;

    for(vi_it = velocity_indices.at((*n_it).first).begin(), i = 0; vi_it != velocity_indices.at((*n_it).first).end(); ++vi_it, ++i)
    {
      for(vj_it = velocity_indices.at((*n_it).first).begin(), j = 0; vj_it != velocity_indices.at((*n_it).first).end(); ++vj_it, ++j) {
        velocity_matrix.add(i, j, vertex_matrix[(*n_it).first][std::make_pair(*vi_it, *vj_it)]);
        if (j != i)
          velocity_matrix.add(j, i, vertex_matrix[(*n_it).first][std::make_pair(*vi_it, *vj_it)]);
      }


      for(p_it = pressure_indices.at((*n_it).first).begin(), j = 0; p_it != pressure_indices.at((*n_it).first).end(); ++p_it, ++j)
        copy_data.pressure_matrix.add(i,j,vertex_matrix[(*n_it).first][std::make_pair(*vi_it, *p_it)]);

      copy_data.velocity_rhs(i) += vertex_rhs.at((*n_it).first)[*vi_it];
    }

    for (p_it = pressure_indices.at((*n_it).first).begin(), i = 0; p_it != pressure_indices.at((*n_it).first).end(); ++p_it, ++i)
      pressure_rhs(i) += vertex_rhs.at((*n_it).first)[*p_it];


    copy_data.Ainverse.reinit(n_edges,n_edges);

    Vector<double> local_pressure_solution;

    Vector<double> tmp_rhs1(n_edges);
    Vector<double> tmp_rhs2(n_edges);
    Vector<double> tmp_rhs3(n_cells);

    // This part applies for both displacement and further stress recovery
    // stress_matrix is SPD so invert it with CG
    copy_data.Ainverse.invert(velocity_matrix);
    copy_data.vertex_pres_matrix.reinit(n_cells, n_cells);
    copy_data.vertex_pres_rhs = pressure_rhs;

    copy_data.vertex_pres_matrix = 0;

    // Computations of LHS=BtAiB
    copy_data.vertex_pres_matrix.triple_product(copy_data.Ainverse, copy_data.pressure_matrix, copy_data.pressure_matrix, true, false);

    copy_data.Ainverse.vmult(tmp_rhs1,copy_data.velocity_rhs,false);
    copy_data.pressure_matrix.Tvmult(tmp_rhs3,tmp_rhs1,false);
    copy_data.vertex_pres_rhs *= -1.0; // -1
    copy_data.vertex_pres_rhs += tmp_rhs3;

    copy_data.p = (*n_it).first;
  }

  template <int dim>
  void MultipointMixedDarcyProblem<dim>::copy_vertex_to_system (const VertexEliminationCopyData &copy_data)
  {
    A_inverse[copy_data.p] = copy_data.Ainverse;
    pressure_matrix[copy_data.p] = copy_data.pressure_matrix;
    velocity_rhs[copy_data.p] = copy_data.velocity_rhs;

    std::set<types::global_dof_index>::iterator pi_it, pj_it;
    unsigned int i, j;
    for (pi_it = pressure_indices[copy_data.p].begin(), i = 0;
         pi_it != pressure_indices[copy_data.p].end();
         ++pi_it, ++i) {
      for (pj_it = pressure_indices[copy_data.p].begin(), j = 0;
           pj_it != pressure_indices[copy_data.p].end();
           ++pj_it, ++j)
        pres_system_matrix.add(*pi_it - n_v, *pj_it - n_v, copy_data.vertex_pres_matrix(i, j));

      pres_rhs(*pi_it - n_v) += copy_data.vertex_pres_rhs(i);
    }
  }

  template <int dim>
  void MultipointMixedDarcyProblem<dim>::pressure_assembly()
  {
    Functions::ParsedFunction<dim> *k_inv = new Functions::ParsedFunction<dim>(dim*dim);
    Functions::ParsedFunction<dim> *bc    = new Functions::ParsedFunction<dim>(1);
    Functions::ParsedFunction<dim> *rhs   = new Functions::ParsedFunction<dim>(1);

    prm.enter_subsection(std::string("permeability ") + Utilities::int_to_string(dim)+std::string("D"));
    k_inv->parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection(std::string("BC ") + Utilities::int_to_string(dim)+std::string("D"));
    bc->parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection(std::string("RHS ") + Utilities::int_to_string(dim)+std::string("D"));
    rhs->parse_parameters(prm);
    prm.leave_subsection();

    KInverse<dim> k_inverse(prm,k_inv);

    TimerOutput::Scope t(computing_timer, "Pressure matrix assembly");

    QGaussLobatto<dim> quad(degree+1);
    QGauss<dim-1> face_quad(degree);

    //pres_system_matrix.reinit(n_p, n_p);
    pres_rhs.reinit(n_p);

    WorkStream::run(vertex_matrix.begin(),
                    vertex_matrix.end(),
                    *this,
                    &MultipointMixedDarcyProblem::vertex_elimination,
                    &MultipointMixedDarcyProblem::copy_vertex_to_system,
                    VertexAssemblyScratchData(fe, triangulation, quad, face_quad,
                                              k_inverse, bc, rhs),
                    VertexEliminationCopyData());

    delete k_inv;
    delete bc;
    delete rhs;

  }

//MultipointMixedDarcyProblem: Solving for velocity
  template <int dim>
  void MultipointMixedDarcyProblem<dim>::velocity_assembly (const typename MapPointMatrix<dim>::iterator &n_it,
                                                            VertexAssemblyScratchData                         &scratch_data,
                                                            VertexEliminationCopyData                         &copy_data)
  {
    unsigned int n_edges = velocity_indices.at((*n_it).first).size();
    unsigned int n_cells = pressure_indices.at((*n_it).first).size();

    Vector<double> local_pressure_solution;
    Vector<double> tmp_rhs1(n_edges);
    Vector<double> tmp_rhs2(n_edges);
    Vector<double> tmp_rhs3(n_cells);

    copy_data.vertex_vel_solution.reinit(n_edges);
    local_pressure_solution.reinit(n_cells);

    std::set<types::global_dof_index>::iterator p_it;
    unsigned int i;

    // RHS computations
    for (p_it = pressure_indices[(*n_it).first].begin(), i = 0; p_it != pressure_indices[(*n_it).first].end(); ++p_it, ++i)
      local_pressure_solution(i) = pres_solution(*p_it - n_v);

    pressure_matrix[(*n_it).first].vmult(tmp_rhs2,local_pressure_solution,false);
    tmp_rhs2 *= -1.0;
    tmp_rhs2+=velocity_rhs[(*n_it).first];
    A_inverse[(*n_it).first].vmult(copy_data.vertex_vel_solution,tmp_rhs2,false);

    copy_data.p = (*n_it).first;
  }

  template <int dim>
  void MultipointMixedDarcyProblem<dim>::copy_vertex_velocity_to_global (const VertexEliminationCopyData &copy_data)
  {
    std::set<types::global_dof_index>::iterator vi_it;
    unsigned int i;

    for (vi_it = velocity_indices[copy_data.p].begin(), i = 0;
         vi_it != velocity_indices[copy_data.p].end();
         ++vi_it, ++i)
    {
      vel_solution(*vi_it) += copy_data.vertex_vel_solution(i);
    }

  }

  template <int dim>
  void MultipointMixedDarcyProblem<dim>::velocity_recovery()
  {
    Functions::ParsedFunction<dim> *k_inv = new Functions::ParsedFunction<dim>(dim*dim);
    Functions::ParsedFunction<dim> *bc    = new Functions::ParsedFunction<dim>(1);
    Functions::ParsedFunction<dim> *rhs   = new Functions::ParsedFunction<dim>(1);

    prm.enter_subsection(std::string("permeability ") + Utilities::int_to_string(dim)+std::string("D"));
    k_inv->parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection(std::string("BC ") + Utilities::int_to_string(dim)+std::string("D"));
    bc->parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection(std::string("RHS ") + Utilities::int_to_string(dim)+std::string("D"));
    rhs->parse_parameters(prm);
    prm.leave_subsection();

    KInverse<dim> k_inverse(prm,k_inv);

    TimerOutput::Scope t(computing_timer, "Velocity solution recovery");

    QGaussLobatto<dim> quad(degree+1);
    QGauss<dim-1> face_quad(degree);

    vel_solution.reinit(n_v);

    WorkStream::run(vertex_matrix.begin(),
                    vertex_matrix.end(),
                    *this,
                    &MultipointMixedDarcyProblem::velocity_assembly,
                    &MultipointMixedDarcyProblem::copy_vertex_velocity_to_global,
                    VertexAssemblyScratchData(fe, triangulation, quad, face_quad,
                                              k_inverse, bc, rhs),
                    VertexEliminationCopyData());

    solution.reinit(2);
    solution.block(0) = vel_solution;
    solution.block(1) = pres_solution;
    solution.collect_sizes();

    delete k_inv;
    delete bc;
    delete rhs;
  }

// MultipointMixedDarcyProblem: Solve SPD cell-centered system for pressure
  template <int dim>
  void MultipointMixedDarcyProblem<dim>::solve_pressure()
  {
    TimerOutput::Scope t(computing_timer, "Pressure CG solve");

    pres_solution.reinit(n_p);

    SolverControl solver_control (10*n_p, 1e-10);
    SolverCG<> solver (solver_control);

    SparseILU<double> preconditioner;
    preconditioner.initialize (pres_system_matrix);

    solver.solve(pres_system_matrix, pres_solution, pres_rhs, preconditioner);
  }


  // MultipointMixedDarcyProblem: run
  template <int dim>
  void MultipointMixedDarcyProblem<dim>::run(const unsigned int refine, const unsigned int grid)
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
      }
      else
        triangulation.refine_global(1);

      vertex_assembly();
      make_cell_centered_sp();
      pressure_assembly();
      solve_pressure ();
      velocity_recovery ();
      postprocess();
      compute_errors (cycle);
      output_results (cycle, refine);
      reset_data_structures ();

      computing_timer.print_summary ();
      computing_timer.reset ();
    }
  }

  template class MultipointMixedDarcyProblem<2>;
  template class MultipointMixedDarcyProblem<3>;
}

