#include "../include/SVDiscrTool.hh"

#include "lwtnn/parse_json.hh"
#include <iostream>
#include <fstream>

void SVDiscrTool::CreateNN(const std::string& json_file_name){

    std::ifstream jsonfile(json_file_name);
    auto config = lwt::parse_json(jsonfile);

    neural_network_ = std::make_unique<const lwt::LightweightNeuralNetwork>(config.inputs, config.layers, config.outputs);

}


std::map<std::string, double> SVDiscrTool::PROB(const Particle& SV){

    lwt::ValueMap var_map{ 
                     
                      {"Evt_pt", SV.Pt()},
                      {"Evt_eta", SV.Eta()},
                      {"Evt_mass", SV.M()},
                      {"Evt_d3d", SV.D3d()},
                      {"Evt_d3dsig", SV.D3dSig()},
                      {"Evt_dxy", SV.Dxy()},
                      {"Evt_costhetaSvPv", SV.CosTheta()},
                      {"Evt_ndof", SV.Ndof()},
                     
        };

    lwt::ValueMap nnout = neural_network_->compute(var_map);

    //std::map<std::string, double> prob_map = nnout->at("flavor");

    return nnout;
}

    

     
