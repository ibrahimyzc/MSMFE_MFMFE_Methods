// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2017 Eldar Khattatov
//
// This file is part of peFLOW.
//
// ---------------------------------------------------------------------
#ifndef PEFLOW_ELASTICITY_MSMFE_H
#define PEFLOW_ELASTICITY_MSMFE_H

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/grid/tria.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/timer.h>

#include "../inc/problem.h"
#include "utilities.h"
#include <unordered_map>

namespace elasticity
{
  using namespace dealii;
  using namespace utilities;
  using namespace peflow;

  /*
   * Class implementing the arbitrary order MSMFE method for
   * Linear Elasticity problem with weak symmetry.
   * Instead of solving the saddle-point type system or using
   * the Schur complement approach we perform local stress
   * and rotation elimination around each node and
   * further use it to assemble the SPD cell-centered
   * displacement system. After solving for displacement the
   * rotation and stress variables are recovered by the same local
   * procedure.
   */
  template <int dim>
  class MultipointMixedElasticityProblem : public ElasticityProblem<dim>
  {
  public:
    using ElasticityProblem<dim>::computing_timer;
    using ElasticityProblem<dim>::fe;
    using ElasticityProblem<dim>::dof_handler;
    using ElasticityProblem<dim>::triangulation;
    using ElasticityProblem<dim>::prm;
    using ElasticityProblem<dim>::degree;
    using ElasticityProblem<dim>::total_dim;
    using ElasticityProblem<dim>::solution;
    using ElasticityProblem<dim>::convergence_table;
    using ElasticityProblem<dim>::compute_errors;
    using ElasticityProblem<dim>::output_results;

    /*
     * Class constructor takes degree and reference to parameter handle
     * as arguments
     */
    MultipointMixedElasticityProblem (const unsigned int degree, ParameterHandler &param);

    /*
     * Main driver function
     */
    virtual void run (const unsigned int refine, const unsigned int grid = 0);
    
    /*
     * Temporary not-tested ugliness
     */
    void set_scaled_rotation(const bool val){ b_scaled_rotation = val; }

  private:
    /*
     * Data structure holding the information needed by threads
     * during assembly process
     */
    struct VertexAssemblyScratchData
    {
      VertexAssemblyScratchData (const FiniteElement<dim> &fe,
                                 const Triangulation<dim> &tria,
                                 const Quadrature<dim> &quad,
                                 const Quadrature<dim-1> &f_quad,
                                 const LameCoefficients<dim> &lame, 
                                 Functions::ParsedFunction<dim> *bc,
                                 Functions::ParsedFunction<dim> *rhs);

      VertexAssemblyScratchData (const VertexAssemblyScratchData &scratch_data);

      FEValues<dim>       fe_values;
      FEFaceValues<dim>   fe_face_values;
      std::vector<int>    n_faces_at_vertex;

      LameCoefficients<dim> lame;
      Functions::ParsedFunction<dim> *bc;
      Functions::ParsedFunction<dim> *rhs;

      const unsigned long num_cells;
    };

    /*
     * Structure to copy data from threads to the main
     */
    struct VertexAssemblyCopyData
    {
      MapPointMatrix<dim>                  cell_mat;
      MapPointVector<dim>                  cell_vec;
      MapPointSet<dim>                     local_displ_indices;
      MapPointSet<dim>                     local_stress_indices;
      MapPointSet<dim>                     local_rotation_indices;
      std::vector<types::global_dof_index> local_dof_indices;
    };

    /*
     * Compute local cell contributions
     */
    void assemble_system_cell (const typename DoFHandler<dim>::active_cell_iterator &cell,
                               VertexAssemblyScratchData                            &scratch_data,
                               VertexAssemblyCopyData                               &copy_data);
    /*
     * Rearrange cell contributions to nodal associated blocks
     */
    struct VertexEliminationCopyData
    {
      // Assembly
      FullMatrix<double> vertex_displ_matrix;
      Vector<double>     vertex_displ_rhs;
      FullMatrix<double> Ainverse;
      FullMatrix<double> CACinverse;
      FullMatrix<double> displacement_matrix;
      FullMatrix<double> rotation_matrix;
      Vector<double>     stress_rhs;
      Vector<double>     rotation_rhs;
      // Recovery
      Vector<double>     vertex_stress_solution;
      Vector<double>     vertex_rotation_solution;
      // Indexing
      Point<dim>         p;
    };
    void copy_cell_to_vertex (const VertexAssemblyCopyData &copy_data);
    void vertex_assembly ();

    /*
     * Assemble and solve displacement cell-centered matrix
     */
    void vertex_elimination (const typename MapPointMatrix<dim>::iterator &n_it,
                             VertexAssemblyScratchData                    &scratch_data,
                             VertexEliminationCopyData                    &copy_data);
    void make_cell_centered_sp ();
    void copy_vertex_to_system (const VertexEliminationCopyData &copy_data);
    void displacement_assembly ();
    void solve_displacement ();

    /*
     * Recover the stress and rotation solutions
     */
    void sr_assembly (const typename MapPointMatrix<dim>::iterator &n_it,
                      VertexAssemblyScratchData                    &scratch_data,
                      VertexEliminationCopyData                    &copy_data);
    void copy_vertex_sr_to_global (const VertexEliminationCopyData &copy_data);
    void sr_recovery ();

    /*
     * Clear all hash tables to start next refinement cycle
     */
    void reset_data_structures ();

    /*
     * Data structures and internal parameters
     */
    SparsityPattern cell_centered_sp;
    SparseMatrix<double> displ_system_matrix;
    Vector<double> displ_rhs;

    std::unordered_map<Point<dim>, FullMatrix<double>, hash_points<dim>, points_equal<dim>> displacement_matrix;
    std::unordered_map<Point<dim>, FullMatrix<double>, hash_points<dim>, points_equal<dim>> rotation_matrix;
    std::unordered_map<Point<dim>, FullMatrix<double>, hash_points<dim>, points_equal<dim>> A_inverse;
    std::unordered_map<Point<dim>, FullMatrix<double>, hash_points<dim>, points_equal<dim>> CAC_inverse;
    std::unordered_map<Point<dim>, Vector<double>, hash_points<dim>, points_equal<dim>> stress_rhs;
    std::unordered_map<Point<dim>, Vector<double>, hash_points<dim>, points_equal<dim>> rotation_rhs;

    MapPointMatrix<dim> vertex_matrix;
    MapPointVector<dim> vertex_rhs;

    MapPointSet<dim> displacement_indices;
    MapPointSet<dim> stress_indices;
    MapPointSet<dim> rotation_indices;

    unsigned long n_s, n_u, n_p;

    Vector<double> displ_solution;
    Vector<double> stress_solution;
    Vector<double> rotation_solution;

    bool b_scaled_rotation;
  };

}

#endif //PEFLOW_ELASTICITY_MSMFE_H
