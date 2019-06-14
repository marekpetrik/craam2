// This file is part of CRAAM, a C++ library for solving plain
// and robust Markov decision processes.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "craam/simulators/inventory.hpp"
#include "utils.hpp"

/**
 * Creates a simple test MDP
 */
//[[Rcpp::export]]
Rcpp::DataFrame mdp_example(Rcpp::String name) {
    return mdp_to_dataframe(create_test_mdp());
}

/**
 * Returns an optional value from a list and a default value if the option is not present.
 */
template <class Type>
inline Type getopt(const Rcpp::List& list, const std::string& name, Type&& defval) {
    return list.containsElementNamed(name.c_str()) ? Rcpp::as<Type>(list[name])
                                                   : std::forward(defval);
}

/**
 * Creates an inventory MDP description
 *
 * @param parameters Parameters describing the inventory management
 *              problem
 */
//[[Rcpp::export]]
Rcpp::DataFrame mdp_inventory(Rcpp::List params) {

    double purchase_cost = Rcpp::as<double>(params["variable_cost"]),
           fixed_cost = Rcpp::as<double>(params["fixed_cost"]),
           holding_cost = Rcpp::as<double>(params["holding_cost"]),
           backlog_cost = Rcpp::as<double>(params["backlog_cost"]),
           sale_price = Rcpp::as<double>(params["sale_price"]);
    long max_inventory = Rcpp::as<long>(params["max_inventory"]),
         max_backlog = Rcpp::as<long>(params["max_backlog"]),
         max_order = Rcpp::as<long>(params["max_order"]);
    craam::numvec demand_probabilities = Rcpp::as<craam::numvec>(params["demands"]);
    long rand_seed = Rcpp::as<long>(params["seed"]);

    std::array<double, 4> costs{purchase_cost, fixed_cost, holding_cost, backlog_cost};
    std::array<long, 3> limits{max_inventory, max_backlog, max_order};

    craam::msen::InventorySimulator simulator(demand_probabilities, costs, sale_price,
                                              limits);
    simulator.set_seed(rand_seed);

    craam::MDP fullmdp;

    simulator.build_mdp(
        [&fullmdp](long statefrom, long action, long stateto, double prob, double rew) {
            add_transition(fullmdp, statefrom, action, stateto, prob, rew);
        });
    return mdp_to_dataframe(fullmdp);
}
