install(
  CODE "MESSAGE(
\"----------------------------------------------------------------------------
CUDOLFINx has now been installed in

    ${CMAKE_INSTALL_PREFIX}


Don't forget to update your environment variables. This can be done
easily using the helper file 'cudolfinx.conf' which sets the appropriate
variables (for users of the Bash shell).

To update your environment variables, run the following command:

    source ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/cudolfinx/cudolfinx.conf

----------------------------------------------------------------------------\")"
)
