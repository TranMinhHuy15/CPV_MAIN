# Handoff RQ3_ABLV3_V6

Đọc README.md và PREREGISTERED_ANALYSIS.md. Đây là code V3matched cho18run mới, không phải prediction V2.
Chạy `make_inputs_from_v2.py`, `launch.py preflight`, `launch.py all`. Base=V6;dùngoutputmới.

Không được bỏA7/G3 hoặc reuseprediction cũ trongmain. Giữc2/loss/bound1/seeds42/43/44. H7A26 làadapter512H26,
khácbaselineLSTM2048longH26. C7khôngthêmquákhứvàoadapternhưngCNN/nềnH7vẫncótemporalcontext.

InitfinaladaptertáchCPU RNG, sharedstatecanonicalgiốngnhauở6arm. Compatibilityadapternguồnkhôngquyếtđịnhfinalweights.
Thaycode/config/plan phảidùngoutputmới; khôngsửamanifestđểépresume. Retentionfalsemặcđịnhgiữfullcheckpoint.
Diagnostics trướccommit, khôngảnhhưởngloss/selection;pilot2→3cầnđượcchạythậttạimáyGPU.

KiểmđịnhmainHolm vàCIvideo3seedfixed;cross-baseexploratorykhôngnhânquả. Khôngchọnnewepoch/loạiseedtheotest.
Các sốV2 chỉ làm độngcơkhámphá,khôngphảikếtquảV3. Đọcverification.jsonđểbiếtcáigìđãtestCPUvàGPUchưatest.
GửivềfolderoutputhoặcZIPcode+log50epoch+config+selectedadapter+prediction/rawcurves/hash,khôngchỉảnh.
