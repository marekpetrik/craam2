// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include "../inst/include/rcraam.h"
#include <Rcpp.h>

using namespace Rcpp;

// worstcase_l1
Rcpp::List worstcase_l1(Rcpp::NumericVector z, Rcpp::NumericVector q, double t);
RcppExport SEXP _rcraam_worstcase_l1(SEXP zSEXP, SEXP qSEXP, SEXP tSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type z(zSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type q(qSEXP);
    Rcpp::traits::input_parameter< double >::type t(tSEXP);
    rcpp_result_gen = Rcpp::wrap(worstcase_l1(z, q, t));
    return rcpp_result_gen;
END_RCPP
}
// pack_actions
Rcpp::List pack_actions(Rcpp::DataFrame mdp);
RcppExport SEXP _rcraam_pack_actions(SEXP mdpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type mdp(mdpSEXP);
    rcpp_result_gen = Rcpp::wrap(pack_actions(mdp));
    return rcpp_result_gen;
END_RCPP
}
// solve_mdp
Rcpp::List solve_mdp(Rcpp::DataFrame mdp, double discount, Rcpp::String algorithm, Rcpp::Nullable<Rcpp::DataFrame> policy_fixed, double maxresidual, size_t iterations, double timeout, bool pack_actions, bool output_tran, bool show_progress);
RcppExport SEXP _rcraam_solve_mdp(SEXP mdpSEXP, SEXP discountSEXP, SEXP algorithmSEXP, SEXP policy_fixedSEXP, SEXP maxresidualSEXP, SEXP iterationsSEXP, SEXP timeoutSEXP, SEXP pack_actionsSEXP, SEXP output_tranSEXP, SEXP show_progressSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type mdp(mdpSEXP);
    Rcpp::traits::input_parameter< double >::type discount(discountSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type algorithm(algorithmSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::DataFrame> >::type policy_fixed(policy_fixedSEXP);
    Rcpp::traits::input_parameter< double >::type maxresidual(maxresidualSEXP);
    Rcpp::traits::input_parameter< size_t >::type iterations(iterationsSEXP);
    Rcpp::traits::input_parameter< double >::type timeout(timeoutSEXP);
    Rcpp::traits::input_parameter< bool >::type pack_actions(pack_actionsSEXP);
    Rcpp::traits::input_parameter< bool >::type output_tran(output_tranSEXP);
    Rcpp::traits::input_parameter< bool >::type show_progress(show_progressSEXP);
    rcpp_result_gen = Rcpp::wrap(solve_mdp(mdp, discount, algorithm, policy_fixed, maxresidual, iterations, timeout, pack_actions, output_tran, show_progress));
    return rcpp_result_gen;
END_RCPP
}
// solve_mdp_rand
Rcpp::List solve_mdp_rand(Rcpp::DataFrame mdp, double discount, Rcpp::String algorithm, Rcpp::Nullable<Rcpp::DataFrame> policy_fixed, double maxresidual, size_t iterations, double timeout, bool output_tran, bool show_progress);
RcppExport SEXP _rcraam_solve_mdp_rand(SEXP mdpSEXP, SEXP discountSEXP, SEXP algorithmSEXP, SEXP policy_fixedSEXP, SEXP maxresidualSEXP, SEXP iterationsSEXP, SEXP timeoutSEXP, SEXP output_tranSEXP, SEXP show_progressSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type mdp(mdpSEXP);
    Rcpp::traits::input_parameter< double >::type discount(discountSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type algorithm(algorithmSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::DataFrame> >::type policy_fixed(policy_fixedSEXP);
    Rcpp::traits::input_parameter< double >::type maxresidual(maxresidualSEXP);
    Rcpp::traits::input_parameter< size_t >::type iterations(iterationsSEXP);
    Rcpp::traits::input_parameter< double >::type timeout(timeoutSEXP);
    Rcpp::traits::input_parameter< bool >::type output_tran(output_tranSEXP);
    Rcpp::traits::input_parameter< bool >::type show_progress(show_progressSEXP);
    rcpp_result_gen = Rcpp::wrap(solve_mdp_rand(mdp, discount, algorithm, policy_fixed, maxresidual, iterations, timeout, output_tran, show_progress));
    return rcpp_result_gen;
END_RCPP
}
// compute_qvalues
Rcpp::DataFrame compute_qvalues(Rcpp::DataFrame mdp, Rcpp::NumericVector valuefunction, double discount);
RcppExport SEXP _rcraam_compute_qvalues(SEXP mdpSEXP, SEXP valuefunctionSEXP, SEXP discountSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type mdp(mdpSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type valuefunction(valuefunctionSEXP);
    Rcpp::traits::input_parameter< double >::type discount(discountSEXP);
    rcpp_result_gen = Rcpp::wrap(compute_qvalues(mdp, valuefunction, discount));
    return rcpp_result_gen;
END_RCPP
}
// rsolve_mdp_sa
Rcpp::List rsolve_mdp_sa(Rcpp::DataFrame mdp, double discount, Rcpp::String nature, SEXP nature_par, Rcpp::String algorithm, Rcpp::Nullable<Rcpp::DataFrame> policy_fixed, double maxresidual, size_t iterations, double timeout, bool pack_actions, bool output_tran, bool show_progress);
RcppExport SEXP _rcraam_rsolve_mdp_sa(SEXP mdpSEXP, SEXP discountSEXP, SEXP natureSEXP, SEXP nature_parSEXP, SEXP algorithmSEXP, SEXP policy_fixedSEXP, SEXP maxresidualSEXP, SEXP iterationsSEXP, SEXP timeoutSEXP, SEXP pack_actionsSEXP, SEXP output_tranSEXP, SEXP show_progressSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type mdp(mdpSEXP);
    Rcpp::traits::input_parameter< double >::type discount(discountSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type nature(natureSEXP);
    Rcpp::traits::input_parameter< SEXP >::type nature_par(nature_parSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type algorithm(algorithmSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::DataFrame> >::type policy_fixed(policy_fixedSEXP);
    Rcpp::traits::input_parameter< double >::type maxresidual(maxresidualSEXP);
    Rcpp::traits::input_parameter< size_t >::type iterations(iterationsSEXP);
    Rcpp::traits::input_parameter< double >::type timeout(timeoutSEXP);
    Rcpp::traits::input_parameter< bool >::type pack_actions(pack_actionsSEXP);
    Rcpp::traits::input_parameter< bool >::type output_tran(output_tranSEXP);
    Rcpp::traits::input_parameter< bool >::type show_progress(show_progressSEXP);
    rcpp_result_gen = Rcpp::wrap(rsolve_mdp_sa(mdp, discount, nature, nature_par, algorithm, policy_fixed, maxresidual, iterations, timeout, pack_actions, output_tran, show_progress));
    return rcpp_result_gen;
END_RCPP
}
// rsolve_mdpo_sa
Rcpp::List rsolve_mdpo_sa(Rcpp::DataFrame mdpo, double discount, Rcpp::String nature, SEXP nature_par, Rcpp::String algorithm, Rcpp::Nullable<Rcpp::DataFrame> policy_fixed, double maxresidual, size_t iterations, double timeout, bool pack_actions, bool output_tran, bool show_progress);
RcppExport SEXP _rcraam_rsolve_mdpo_sa(SEXP mdpoSEXP, SEXP discountSEXP, SEXP natureSEXP, SEXP nature_parSEXP, SEXP algorithmSEXP, SEXP policy_fixedSEXP, SEXP maxresidualSEXP, SEXP iterationsSEXP, SEXP timeoutSEXP, SEXP pack_actionsSEXP, SEXP output_tranSEXP, SEXP show_progressSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type mdpo(mdpoSEXP);
    Rcpp::traits::input_parameter< double >::type discount(discountSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type nature(natureSEXP);
    Rcpp::traits::input_parameter< SEXP >::type nature_par(nature_parSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type algorithm(algorithmSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::DataFrame> >::type policy_fixed(policy_fixedSEXP);
    Rcpp::traits::input_parameter< double >::type maxresidual(maxresidualSEXP);
    Rcpp::traits::input_parameter< size_t >::type iterations(iterationsSEXP);
    Rcpp::traits::input_parameter< double >::type timeout(timeoutSEXP);
    Rcpp::traits::input_parameter< bool >::type pack_actions(pack_actionsSEXP);
    Rcpp::traits::input_parameter< bool >::type output_tran(output_tranSEXP);
    Rcpp::traits::input_parameter< bool >::type show_progress(show_progressSEXP);
    rcpp_result_gen = Rcpp::wrap(rsolve_mdpo_sa(mdpo, discount, nature, nature_par, algorithm, policy_fixed, maxresidual, iterations, timeout, pack_actions, output_tran, show_progress));
    return rcpp_result_gen;
END_RCPP
}
// rsolve_mdp_s
Rcpp::List rsolve_mdp_s(Rcpp::DataFrame mdp, double discount, Rcpp::String nature, SEXP nature_par, Rcpp::String algorithm, Rcpp::Nullable<Rcpp::DataFrame> policy_fixed, double maxresidual, size_t iterations, double timeout, bool pack_actions, bool output_tran, bool show_progress);
RcppExport SEXP _rcraam_rsolve_mdp_s(SEXP mdpSEXP, SEXP discountSEXP, SEXP natureSEXP, SEXP nature_parSEXP, SEXP algorithmSEXP, SEXP policy_fixedSEXP, SEXP maxresidualSEXP, SEXP iterationsSEXP, SEXP timeoutSEXP, SEXP pack_actionsSEXP, SEXP output_tranSEXP, SEXP show_progressSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type mdp(mdpSEXP);
    Rcpp::traits::input_parameter< double >::type discount(discountSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type nature(natureSEXP);
    Rcpp::traits::input_parameter< SEXP >::type nature_par(nature_parSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type algorithm(algorithmSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::DataFrame> >::type policy_fixed(policy_fixedSEXP);
    Rcpp::traits::input_parameter< double >::type maxresidual(maxresidualSEXP);
    Rcpp::traits::input_parameter< size_t >::type iterations(iterationsSEXP);
    Rcpp::traits::input_parameter< double >::type timeout(timeoutSEXP);
    Rcpp::traits::input_parameter< bool >::type pack_actions(pack_actionsSEXP);
    Rcpp::traits::input_parameter< bool >::type output_tran(output_tranSEXP);
    Rcpp::traits::input_parameter< bool >::type show_progress(show_progressSEXP);
    rcpp_result_gen = Rcpp::wrap(rsolve_mdp_s(mdp, discount, nature, nature_par, algorithm, policy_fixed, maxresidual, iterations, timeout, pack_actions, output_tran, show_progress));
    return rcpp_result_gen;
END_RCPP
}
// set_rcraam_threads
void set_rcraam_threads(int n);
RcppExport SEXP _rcraam_set_rcraam_threads(SEXP nSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    set_rcraam_threads(n);
    return R_NilValue;
END_RCPP
}
// mdp_from_samples
Rcpp::DataFrame mdp_from_samples(Rcpp::DataFrame samples_frame);
RcppExport SEXP _rcraam_mdp_from_samples(SEXP samples_frameSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type samples_frame(samples_frameSEXP);
    rcpp_result_gen = Rcpp::wrap(mdp_from_samples(samples_frame));
    return rcpp_result_gen;
END_RCPP
}
// mdp_example
Rcpp::DataFrame mdp_example(Rcpp::String name);
RcppExport SEXP _rcraam_mdp_example(SEXP nameSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::String >::type name(nameSEXP);
    rcpp_result_gen = Rcpp::wrap(mdp_example(name));
    return rcpp_result_gen;
END_RCPP
}
// mdp_inventory
Rcpp::DataFrame mdp_inventory(Rcpp::List params);
RcppExport SEXP _rcraam_mdp_inventory(SEXP paramsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type params(paramsSEXP);
    rcpp_result_gen = Rcpp::wrap(mdp_inventory(params));
    return rcpp_result_gen;
END_RCPP
}
// mdp_population
Rcpp::DataFrame mdp_population(int capacity, int initial, Rcpp::NumericMatrix growth_rates_exp, Rcpp::NumericMatrix growth_rates_std, Rcpp::NumericMatrix rewards, double external_mean, double external_std, Rcpp::String s_growth_model);
RcppExport SEXP _rcraam_mdp_population(SEXP capacitySEXP, SEXP initialSEXP, SEXP growth_rates_expSEXP, SEXP growth_rates_stdSEXP, SEXP rewardsSEXP, SEXP external_meanSEXP, SEXP external_stdSEXP, SEXP s_growth_modelSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type capacity(capacitySEXP);
    Rcpp::traits::input_parameter< int >::type initial(initialSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type growth_rates_exp(growth_rates_expSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type growth_rates_std(growth_rates_stdSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type rewards(rewardsSEXP);
    Rcpp::traits::input_parameter< double >::type external_mean(external_meanSEXP);
    Rcpp::traits::input_parameter< double >::type external_std(external_stdSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type s_growth_model(s_growth_modelSEXP);
    rcpp_result_gen = Rcpp::wrap(mdp_population(capacity, initial, growth_rates_exp, growth_rates_std, rewards, external_mean, external_std, s_growth_model));
    return rcpp_result_gen;
END_RCPP
}
// simulate_mdp
Rcpp::DataFrame simulate_mdp(Rcpp::DataFrame mdp, int initial_state, Rcpp::DataFrame policy, int horizon, int episodes);
RcppExport SEXP _rcraam_simulate_mdp(SEXP mdpSEXP, SEXP initial_stateSEXP, SEXP policySEXP, SEXP horizonSEXP, SEXP episodesSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type mdp(mdpSEXP);
    Rcpp::traits::input_parameter< int >::type initial_state(initial_stateSEXP);
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type policy(policySEXP);
    Rcpp::traits::input_parameter< int >::type horizon(horizonSEXP);
    Rcpp::traits::input_parameter< int >::type episodes(episodesSEXP);
    rcpp_result_gen = Rcpp::wrap(simulate_mdp(mdp, initial_state, policy, horizon, episodes));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_rcraam_worstcase_l1", (DL_FUNC) &_rcraam_worstcase_l1, 3},
    {"_rcraam_pack_actions", (DL_FUNC) &_rcraam_pack_actions, 1},
    {"_rcraam_solve_mdp", (DL_FUNC) &_rcraam_solve_mdp, 10},
    {"_rcraam_solve_mdp_rand", (DL_FUNC) &_rcraam_solve_mdp_rand, 9},
    {"_rcraam_compute_qvalues", (DL_FUNC) &_rcraam_compute_qvalues, 3},
    {"_rcraam_rsolve_mdp_sa", (DL_FUNC) &_rcraam_rsolve_mdp_sa, 12},
    {"_rcraam_rsolve_mdpo_sa", (DL_FUNC) &_rcraam_rsolve_mdpo_sa, 12},
    {"_rcraam_rsolve_mdp_s", (DL_FUNC) &_rcraam_rsolve_mdp_s, 12},
    {"_rcraam_set_rcraam_threads", (DL_FUNC) &_rcraam_set_rcraam_threads, 1},
    {"_rcraam_mdp_from_samples", (DL_FUNC) &_rcraam_mdp_from_samples, 1},
    {"_rcraam_mdp_example", (DL_FUNC) &_rcraam_mdp_example, 1},
    {"_rcraam_mdp_inventory", (DL_FUNC) &_rcraam_mdp_inventory, 1},
    {"_rcraam_mdp_population", (DL_FUNC) &_rcraam_mdp_population, 8},
    {"_rcraam_simulate_mdp", (DL_FUNC) &_rcraam_simulate_mdp, 5},
    {NULL, NULL, 0}
};

RcppExport void R_init_rcraam(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
