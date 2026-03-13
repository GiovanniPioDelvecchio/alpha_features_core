find_package(pybind11 REQUIRED)

add_library(alpha_features_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/alpha_features_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/alpha_features_core.h
)

pybind11_add_module(alpha_features_core
    ${CMAKE_CURRENT_SOURCE_DIR}/bindings.cpp
)

set_target_properties(alpha_features_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(alpha_features_core PRIVATE alpha_features_lib)
target_include_directories(alpha_features_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS alpha_features_core DESTINATION alpha_features_core)