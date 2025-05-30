

#include "base_predictor.h"

#include "tage/tage.h"
#include "tage/tage_scl.h"
#include "llbp/llbp.h"
#include "utils/json.hpp"


BasePredictor* CreateBP(std::string bp_name)
{
    if (bp_name == "tage64k") {
        return new LLBP::Tage64k();
    } else if (bp_name == "tage64kscl") {
        return new LLBP::TageSCL64k();
    } else if (bp_name == "tage512kscl") {
        return new LLBP::TageSCL512k();
    } else if (bp_name == "llbp") {
        return new LLBP::LLBPTageSCL64k();
    } else if (bp_name == "llbp-timing") {
        return new LLBP::LLBPTageSCL64kTiming();
    } else {
        std::cout << "Wrong BP name: " << bp_name << std::endl;
        exit(EXIT_FAILURE);
    }
}

// BasePredictor* CreateBP(std::string bp_name, LLBP::LLBPConfig conf)
BasePredictor* CreateBP(std::string bp_name, nlohmann::json json_conf)
{

    LLBP::LLBPConfig conf;
    LLBP::from_json(json_conf, conf);

    // if (bp_name == "tage64k") {
    //     return new LLBP::Tage64k();
    // } else if (bp_name == "tage64kscl") {
    //     return new LLBP::TageSCL64k();
    // } else if (bp_name == "tage512kscl") {
    //     return new LLBP::TageSCL512k();
    // } else 
    if (bp_name == "llbp") {
        return new LLBP::LLBPTageSCL64k(conf);
    }
    else if (bp_name == "llbp-timing") {
        return new LLBP::LLBPTageSCL64kTiming();
    }
    // } else {
    //     std::cout << "Wrong BP name: " << bp_name << std::endl;
    //     exit(EXIT_FAILURE);
    // }
}
