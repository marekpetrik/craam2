#pragma once 
#define VERSION 3.0.0

// make sure that these settings can be defined outside
#ifndef CRAAM_CONFIG_HPP

/* #undef IS_DEBUG */
/* #undef GUROBI_USE */

#ifndef IS_DEBUG
    #define NDEBUG
#endif

#endif // CRAAM_CONFIG_HPP




