#pragma once 
#define VERSION 3.0.0

// make sure that these settings can be defined outside
#ifndef CRAAM_CONFIG_HPP
#define CRAAM_CONFIG_HPP

/* #undef IS_DEBUG */
#define GUROBI_USE

#ifndef IS_DEBUG
    #define NDEBUG
#endif

#endif // CRAAM_CONFIG_HPP




