!include nsDialogs.nsh
!include LogicLib.nsh

Var EsmLicenseCheckbox
Var EsmLicenseAccepted

!macro NSIS_HOOK_PREINSTALL
  IfSilent protcross_silent_license_check protcross_interactive_license_check
protcross_silent_license_check:
  ReadEnvStr $0 "PROTCROSS_DESKTOP_CI_ACCEPT_ESMC_LICENSE"
  StrCmp $0 "1" protcross_license_confirmed 0
  Abort
protcross_interactive_license_check:
  MessageBox MB_ICONQUESTION|MB_YESNO "ProtCross Desktop prediction support requires ESM-C weights. The weights are not included in this installer. By continuing, you confirm that you have reviewed and agree to comply with the applicable ESM-C model license before downloading or importing ESM-C weights." IDYES protcross_license_confirmed
  Abort
protcross_license_confirmed:
!macroend

Function ProtCrossEsmLicensePageCreate
  nsDialogs::Create 1018
  Pop $0
  ${If} $0 == error
    Abort
  ${EndIf}

  ${NSD_CreateLabel} 0 0 100% 52u "ProtCross Desktop uses ESM-C weights for residue embeddings. These weights are not redistributed in this installer. You must review and agree to the applicable ESM-C model license before configuring prediction assets."
  Pop $1

  ${NSD_CreateLink} 0 58u 100% 12u "Open ESM-C license: https://www.evolutionaryscale.ai/policies/cambrian-non-commercial-license-agreement"
  Pop $2

  ${NSD_CreateCheckbox} 0 82u 100% 14u "I have reviewed and agree to comply with the applicable ESM-C model license."
  Pop $EsmLicenseCheckbox
  ${NSD_Uncheck} $EsmLicenseCheckbox

  nsDialogs::Show
FunctionEnd

Function ProtCrossEsmLicensePageLeave
  ${NSD_GetState} $EsmLicenseCheckbox $EsmLicenseAccepted
  ${If} $EsmLicenseAccepted != ${BST_CHECKED}
    MessageBox MB_ICONEXCLAMATION "You must confirm the ESM-C model license before installing ProtCross Desktop prediction support."
    Abort
  ${EndIf}
FunctionEnd
