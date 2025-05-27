// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2017 Eldar Khattatov
//
// This file is part of peFLOW.
//
// ---------------------------------------------------------------------

#ifndef PEFLOW_DARCY_MFMFE_H
#define PEFLOW_DARCY_MFMFE_H

#include <deal.II/base/parsed_function.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_dgq.h>

#include <unordered_map>

#include "../inc/problem.h"
#include "../inc/utilities.h"
#include "../inc/darcy_data.h"

namespace darcy
{
  using namespace dealii;
  using namespace utilities;
  using namespace peflow;

  /*
   * Class implementing the arbitrary order MFMFE method for
   * Darcy problem. Instead of solving the saddle-point type
   * system or using the Schur complement approach we
   * perform local velocity elimination around each node and
   * further use it to assemble the SPD cell-centered
   * pressure system. After solving for pressure the velocity
   * is recovered by the same local procedure.
   */
  template <int dim>
  class MultipointMixedDarcyProblem : public DarcyProblem<dim>
  {
  public:
    using DarcyProblem<dim>::computing_timer;
    using DarcyProblem<dim>::fe;
    using DarcyProblem<dim>::dof_handler;
    using DarcyProblem<dim>::triangulation;
    using DarcyProblem<dim>::prm;
    using DarcyProblem<dim>::degree;
    using DarcyProblem<dim>::solution;
    using DarcyProblem<dim>::convergence_table;
    using DarcyProblem<dim>::postprocess;
    using DarcyProblem<dim>::compute_errors;
    using DarcyProblem<dim>::output_results;
    /*
     * Class constructor takes degree and reference to parameter handle
     * as arguments
     */
    MultipointMixedDarcyProblem (const unsigned int degree, ParameterHandler &);

    /*
     * Main driver function
     */
    void run (const unsigned int refine, const unsigned int grid = 0);
  private:
    /*
     * Data structure holding the information needed by threads
     * during assembly process
     */
    struct VertexAssemblyScratchData
    {
      VertexAssemblyScratchData (const FiniteElement<dim> &fe,
                                 const Triangulation<dim>       &tria,
                                 const Quadrature<dim> &quad,
                                 const Quadrature<dim-1> &f_quad,
                                 const KInverse<dim> &k_data, 
                                 Functions::ParsedFunction<dim> *bc,
                                 Functions::ParsedFunction<dim> *rhs);

      VertexAssemblyScratchData (const VertexAssemblyScratchData &scratch_data);

      FEValues<dim>       fe_values;
      FEFaceValues<dim>   fe_face_values;
      std::vector<int>    n_faces_at_vertex;

      KInverse<dim>     K_inv;
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
      MapPointSet<dim>                     local_pres_indices;
      MapPointSet<dim>                     local_vel_indices;
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
      FullMatrix<double> vertex_pres_matrix;
      Vector<double>     vertex_pres_rhs;
      FullMatrix<double> Ainverse;
      FullMatrix<double> pressure_matrix;
      Vector<double>     velocity_rhs;
      // Recovery
      Vector<double>     vertex_vel_solution;
      // Indexing
      Point<dim>         p;
    };

    void copy_cell_to_vertex (const VertexAssemblyCopyData &copy_data);
    void vertex_assembly ();

    /*
     * Assemble and solve pressure cell-centered matrix
     */
    void make_cell_centered_sp ();
    void vertex_elimination (const typename MapPointMatrix<dim>::iterator &n_it,
                             VertexAssemblyScratchData                    &scratch_data,
                             VertexEliminationCopyData                    &copy_data);
    void copy_vertex_to_system (const VertexEliminationCopyData                            &copy_data);
    void pressure_assembly ();
    void solve_pressure ();

    /*
     * Recover the velocity solution
     */
    void velocity_assembly (const typename MapPointMatrix<dim>::iterator &n_it,
                            VertexAssemblyScratchData                  &scratch_data,
                            VertexEliminationCopyData                  &copy_data);
    void copy_vertex_velocity_to_global (const VertexEliminationCopyData &copy_data);
    void velocity_recovery ();

    /*
     * Clear all hash tables to start next refinement cycle
     */
    void reset_data_structures ();

    /*
     * Data structures and internal parameters
     */
    SparsityPattern cell_centered_sp;
    SparseMatrix<double> pres_system_matrix;
    Vector<double> pres_rhs;

    std::unordered_map<Point<dim>, FullMatrix<double>, hash_points<dim>, points_equal<dim>> pressure_matrix;
    std::unordered_map<Point<dim>, FullMatrix<double>, hash_points<dim>, points_equal<dim>> A_inverse;
    std::unordered_map<Point<dim>, Vector<double>, hash_points<dim>, points_equal<dim>> velocity_rhs;

    MapPointMatrix<dim> vertex_matrix;
    MapPointVector<dim> vertex_rhs;

    MapPointSet<dim> pressure_indices;
    MapPointSet<dim> velocity_indices;

    unsigned long n_v, n_p;

    Vector<double> pres_solution;
    Vector<double> vel_solution;
  };

}

#endif //PEFLOW_DARCY_MFMFE_H
