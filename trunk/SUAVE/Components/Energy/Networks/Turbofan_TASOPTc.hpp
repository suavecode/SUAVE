#include <stdio.h>
#include <string.h>
#include <iostream>
#include <string>
#include <cmath>
#include <armadillo>
#include <fstream>


class engine_parameters{
    
    public:
    
        double aalpha,pi_d,eta_d,pi_f,eta_f,pi_fn,eta_fn,pi_lc,eta_lc,pi_hc,eta_hc,Tt4,pi_b,eta_b,htf;
        double eta_lt,etam_lt,eta_ht,etam_ht,pi_tn,eta_tn,Pt5,M2,M2_5,HTR_f,HTR_hc,Tref,Pref,GF,N_spec,Tt4_Tt2;
        double mf,mlc,mhc,Nln,Nhn,Nfn,mdot_core;
        double A5,A7,A2,A2_5,mhtD,mltD,mhcD,mlcD,mfD,NlcD,NhcD,NlD,NhD,Tt3,Pt3,Nl,Nh,Nf,Tt4_spec;
        double Res[8];
        double norm_R;
        double d_fan;
    
    
    //engine_parameters()
    
};




class Turbofan_TASOPTc{
    
    
    public:
    
        int number_of_engines,cooling_flow,max_iters;
        double newton_relaxation,norm_R,total_design_thrust;
        engine_parameters *dp,*odp_b,*dp_temp;
        double Res[8],Jac[8][8],bp[8]; //d_bp[8];
        double Thrust_fin, sfc_fin, mdot_fin;
        double sls_thrust,sls_sfc;
        double fan_diameter;
    
    
    

    
        Turbofan_TASOPTc(void);
        void unpack();
        void unpack2(double *eng_params);
        void evaluate(double M0,double P0,double T0,double throttle,int flag,double * outputs, engine_parameters * odp);
        void size(double Ms,double Ps,double Ts,double throttle,double design_thrust,int no_engines);
        void offdesign(double M,double P,double T,double throttle);
        //void set_tem_baseline_params(bp,d_a1,d_bp);
        void set_baseline_params(double *bp);
        void update_baseline_params(double *bp);
        void reset_baseline_params(void);
        void performance(double pid,double mdot,int flag,double *performance_values);
        void jacobian_fd(double *bp, double M, double P, double T, double throttle);
        double Tt(double M,double T,double gamma);
        double Pt(double M,double P,double gamma);
        double Tt_inv(double M,double Tt,double gamma);
        double Pt_inv(double M,double Pt,double gamma);
    
//        void jacobian_an(double gamma,double R,double Cp,double P0,double T0,double M0,double throttle,double dp_pilc,double dp_mlc,double dp_pi_hc,double dp_mhc,double dp_pi_f,double dp_mf,double dp_pi_d,double dp_pi_fn,double dp_eta_fn,double dp_pi_lc,double dp_Tt4,double dp_eta_b,double dp_htf,double dp_pi_b,double dp_Pref,double dp_Tref,double dp_etam_ht,double dp_eta_ht,double dp_etam_lt,double dp_aalpha,double dp_eta_lt,double dp_pi_tn,double dp_Pt5,double dp_Tt4_spec,double dp_Tt4_Tt2,double dp_M2,double dp_M2_5,double dp_GF,double dp_mhtD,double dp_mltD,double dp_A7,double dp_A5,double dp_N_spec,double Nln,double Nhn,double Nfn,double dp_eta_lc,double dp_eta_hc,double dp_eta_f,double M7,double P7,double M5,double P5,double dNln_pilc,double dNln_mlc,double dNfn_pif,double dNfn_mf);
    
    void jacobian_an(double gamma,double R,double Cp,double P0,double T0,double M0,double throttle,double Nln,double Nhn,double Nfn,double M7,double P7,double M5,double P5,double dNln_pilc,double dNln_mlc,double dNfn_pif,double dNfn_mf,double M6, double M8);
    
    //void jacobian_an();
};



