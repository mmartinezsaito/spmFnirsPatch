# spmFnirsPatch
Code snippet to augment the SPM-fNIRS toolbox (Tak et al., 2016) with parametric modulators and automatic looping over subjects.
The code is provided as is, and sparsely commented. 

## Instructions
1. Download the SPM-fNIRS toolbox.
2. Substitute the appropriate m-files with the replacement files.
3. Run the main file.

## Files
- Main file
  * darnirs.m
- Replacement files
  * spm_fnirs_read_nirscout.m 
  * spm_fnirs_convert_ui.m
  * spm_fnirs_temporalpreproc_ui.m
  * spm_fnirs_con_2d.m
  * spm_fnirs_con_3d.m
 
## References
- Tak S, Uga M, Flandin G, Dan I, Penny WD (2016) Sensor space group analysis for fNIRS data. J Neurosci Meth, 264:103–112
