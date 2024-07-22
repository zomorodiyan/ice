cmake_minimum_required(VERSION 3.16.3)

set(This ExampleTest)

set(Sources
  ExampleTest.cpp
)

add_executable(${This} ${Sources})
target_link_libraries(${This} PUBLIC
  gtest_main
  orange
)

add_test(
  NAME ${This}
  COMMAND ${This}
)
