Below is my full research publication's LaTeX document.

\begin{document}

\title{Learning Where to Sense: Efficient Arsenic Hotspot Localization Through Adaptive Sensor Placement}

\author{\IEEEauthorblockN{Aditya Shivakumar}
\IEEEauthorblockA{The Harker High School, San Jose, CA \\
\{adityashivakumar7@gmail.com\}}
}

\maketitle

\section{Introduction}

Arsenic contamination in drinking water affects over 500 million people worldwide, leading to severe health outcomes including skin lesions, cardiovascular disease, diabetes, and multiple cancers~\cite{fatoki2022arsenic, kuo2022association}. This issue disproportionately impacts developing regions in South Asia~\cite{rasheed2017human}. Children face the largest risk: those under three consume the highest levels relative to body weight, largely through rice-based diets~\cite{efsa2009panel}. Early-life exposure has been associated with a 14-fold increase in liver cancer, a 50-fold increase in bronchiectasis, and measurable IQ loss in these children~\cite{naujokas2013broad, hamadani2011critical}.

Inorganic arsenic primarily exists in two states: arsenite (As(III)) and arsenate (As(V)). Arsenic speciation depends on redox conditions: As(III) tends to dominate in anoxic environments, such as shallow groundwater in Bangladesh, whereas As(V) is more common under oxidizing conditions, such as those found in India and Pakistan~\cite{rasheed2017human}.

\subsection{Related Work}
Current methods for detecting arsenic in water, such as Inductively Coupled Plasma Mass Spectrometry (ICP-MS), Atomic Absorption Spectroscopy (AAS), and Surface-Enhanced Raman Spectroscopy (SERS), provide high sensitivity and reproducibility. However, equipment costs of \$10,000–\$50,000, combined with the need for trained staff and laboratory facilities, limit their use in low-resource settings.

Nanoparticle-based colorimetric assays offer a simpler and more affordable alternative. These assays use aggregation-induced shifts in the localized surface plasmon resonance (LSPR) of gold or silver nanoparticles~\cite{chang2019gold}, producing a visible red-to-blue color change. This approach has been demonstrated for As(III)\cite{Boruah2019Green,Boruah2019Silver,Shrivas2015,Divsar2015,Gupta2016,Gong2017} and As(V)\cite{Wi2023,Das2016,Liu2015,Liu2014,Lace2019}. Because As(III) has been more extensively studied, we focus on developing a low-cost detection method for As(V), a growing concern in India and Pakistan~\cite{rasheed2017human}.

\cite{wi2023highly} demonstrated that methylene-blue-functionalized AuNPs can selectively detect As(V) through ion-pairing interactions under mildly alkaline conditions. Their assay achieved a clear linear dynamic range (LDR), enabling straightforward quantification from absorbance ratios. However, it relies on high-purity, custom-synthesized nanoparticles using the modified Frens method. The nanoparticles last only 6 days at room temperature and 20 days at \SI{4}{\celsius}~\cite{yang2024review}, making them prohibitively expensive, impractical for field deployment, and limited to laboratory use.

In this study, we employ commercially available citrate-capped AuNPs instead of lab-synthesized ones, adapting their chemistry for field deployment. The stabilized nanoparticles offer a 12-month shelf life~\cite{SigmaAldrich900474} and substantially lower costs.

\subsection{Contributions}
This work makes four key contributions:
\begin{enumerate}
    \item A glassware-based MB--AuNP assay that leverages premade AuNPs and dialysis to lower costs.
    \item UV--Vis datasets collected at sparse concentrations and used to learn spectral behavior.
    \item A field-ready, phone-based workflow for quantitative arsenic sensing.
\end{enumerate}

\section{Methods: Low-Cost Methylene Blue Sensor Synthesis}

\subsection{Reagents and Instrumentation}

 We use ultrapure Milli-Q water (resistivity \SI{18.2}{\mega\ohm\centi\meter}) for all dilutions and preparations, ensuring minimal background contamination. Citrate-capped gold nanoparticles (AuNPs, 15~nm diameter, OD 1; Sigma-Aldrich Cat. No. 777137) act as the optical transduction element. Standardized arsenate [As(V)] solutions from Inorganic Ventures provide the intermediate and working stocks. Methylene blue trihydrate (purity $\geq$95\%; Thermo Scientific Chemicals Cat. No. AC126751000) functionalizes the AuNPs and introduces the cationic binding sites required for arsenate detection.  

We record absorbance spectra with a Horiba Aqualog UV--Vis spectrophotometer across 200–799~nm with 1~nm resolution. We quantify dissolved organic carbon (DOC) during dialysis using a Shimadzu DOC analyzer, with citrate removal serving as the indicator of stabilizer clearance. We carry out dialysis with Pur-A-Lyzer Mega units (1~kDa MWCO, Sigma-Aldrich). These units reduce cost, simplify setup, and allow reuse for hundreds to thousands of preparations. In contrast to traditional dialysis bags, they avoid leaks and cumbersome assembly, making them more reliable for on-site field use. We monitor pH with an Orion Versa Star Pro pH meter (Thermo Scientific, VSTAR13 Difficult Samples Kit) to ensure consistent speciation of arsenate during assays.  

\subsection{Arsenate Preparation and AuNP Dialysis}

We prepare arsenate working solutions by serially diluting a \SI{1000}{\micro\gram\per\milli\liter} arsenate standard to a \SI{10000}{ppb} intermediate stock in a high-density polyethylene (HDPE) container. From this stock, we produce a \SI{500}{ppb} working solution. We adjust the working solution to pH~9.0 with 0.01~M \ce{NaOH}, following~\cite{wi2023highly}. At this pH, HAsO$_4^{2-}$ dominates the speciation profile of arsenic acid. Because HAsO$_4^{2-}$ forms ion pairs with MB$^+$, maintaining pH~9.0 maximizes assay sensitivity.  

The as-purchased AuNPs introduce a complication: free citrate and proprietary stabilizers adsorb onto the nanoparticle surface, blocking reactive groups and suppressing aggregation. Wi and Kim~\cite{wi2023highly} avoid this problem by synthesizing high-purity AuNPs through the Frens method, but that route is expensive and produces suspensions with limited shelf life. To work with commercial AuNPs, we instead adopt a pretreatment step unique to this study: dialysis to remove citrate and stabilizers. Gong et al.~\cite{gong2017colorimetric} establish precedent for this approach by dialyzing citrate-capped AuNPs for an As(III) sensing assay.  

For dialysis, we load \SI{10}{\milli\liter} of AuNP suspension into a 1~kDa MWCO Pur-A-Lyzer unit. We pre-soak the unit in Milli-Q water to confirm seal integrity, then immerse it in \SI{300}{\milli\liter} of Milli-Q water in a sealed glass flask under magnetic stirring. We replace the external bath at intervals over a 40-hour period. To monitor stabilizer removal, we collect \SI{40}{\milli\liter} aliquots of the dialysate at 16, 22, and 40 hours and analyze them for DOC. Each citrate molecule contributes six carbons, so increases in DOC provide a direct measure of citrate and organic stabilizer diffusion into the bath.  

After dialysis, we dilute the AuNP suspension 2.5-fold and readjust the pH to 9.0 with 0.1~M \ce{NaOH}. This produces citrate-free AuNPs at the appropriate surface chemistry and solution conditions for subsequent functionalization with methylene blue.  

\subsection{UV--Vis Assay Protocols and Baseline Controls}

Before functionalization, we establish baseline behavior and confirm selectivity. Quartz cuvettes are rigorously cleaned before each assay: we soak them in methanol, rinse them 80 times with Milli-Q water, soak them again in Milli-Q for 15 minutes, and perform a final 20 rinses. We then collect baseline spectra of the dialyzed AuNPs in triplicate.  

To test selectivity, we add \SI{20}{ppb} As(III) to dialyzed AuNPs and incubate the mixture for 10 minutes before scanning. We observe no significant red-to-blue shift in the plasmon band, verifying that the aggregation response is specific to As(V) under these conditions.  

We also evaluate an alternative purification strategy based on centrifugation to determine whether residual stabilizers suppress aggregation. We centrifuge \SI{1.5}{\milli\liter} aliquots of AuNP suspension at 17,000~g for 15 minutes, discard the supernatant, and resuspend the pellet in fresh Milli-Q. We repeat this cycle three times. Although centrifugation removes stabilizers, it also induces over-aggregation and destabilization of samples. The method is prohibitively expensive at scale. We therefore favor dialysis as the more reproducible and practical method.  

\subsection{MB Functionalization, Optimization, and As(V) Detection}

We prepare a \num{1e-3}~\si{\Molar} MB stock in Milli-Q and dilute it to \num{1e-5}~\si{\Molar} for functionalization. Because the AuNP concentration in this study (\SI{22.4}{\ppm}) is substantially lower than that used by Wi and Kim (\SI{105}{\ppm})~\cite{wi2023highly}, we adjust MB:AuNP ratios proportionally.  

We frame functionalization as an optimization problem. Too little MB provides insufficient cationic binding sites, producing weak ion-pairing with HAsO$_4^{2-}$ and minimal aggregation in response to As(V). Too much MB induces nonspecific aggregation, broadening spectra and shifting the SPR band even in the absence of arsenate. The objective is to identify a formulation that produces a strong As(V)-induced response while maintaining colloidal stability without analyte present.  

We optimize in two phases. In a \textbf{coarse sweep}, we mix \SI{2.5}{\milli\liter} of AuNP suspension with MB volumes of \SI{0.10}{\milli\liter}, \SI{0.30}{\milli\liter}, \SI{0.60}{\milli\liter}, and \SI{0.90}{\milli\liter}. We gently invert samples, incubate them for 30 minutes at room temperature to allow surface adsorption~\cite{salimi2018colorimetric}, and then scan them. Below \SI{0.10}{\milli\liter}, we observe no measurable aggregation with As(V). Above \SI{0.30}{\milli\liter}, we observe irreversible aggregation even before adding analyte. Thus, in a \textbf{fine titration}, we focus on the 0.10--\SI{0.30}{\milli\liter} window and vary MB volumes in 0.01~mL increments. 

From this process, we identify two candidate formulations: AuNPs@MB-0.115 and AuNPs@MB-0.30. To evaluate performance, we add the prepared As(V) stock dropwise to reach target concentrations and dividing them into micro-aliquots. After each addition, we invert samples, incubate them for 10 minutes to allow MB$^+$--HAsO$_4^{2-}$ ion-pairing, and scan them. We quantify aggregation by measuring absorbance changes in the 620--660~nm region, corresponding to the red-to-blue shift from nanoparticle clustering.  

We ultimately select AuNPs@MB-0.30 for its stronger colorimetric response. The UV--Vis scans at each target concentration generate $N$ \textbf{anchor spectra}, where the spectrum at $c_i$ consists of $L$ wavelength--absorbance pairs $(\lambda_j, A_{ij})$. Since measurements span 200--799~nm with 1~nm resolution, each spectrum contains $L=600$ samples. These anchor spectra form the dataset used for subsequent modeling.  

\subsection{Spectral Interpolation}

Our goal is to interpolate spectra between anchor measurements. A phone camera needs a continuous gradient to map subtle shade differences to concentration. This requires modeling 
\((c,\lambda) \mapsto A(c,\lambda)\), so that intermediate absorbance values can be reconstructed at concentrations not directly measured.  

Because absorbance generally changes smoothly with concentration, we adopt a straightforward linear interpolation strategy. For each wavelength \(\lambda\), the absorbance at an unmeasured concentration \(c\) is estimated by connecting the two nearest anchor points with a straight line. If spectra are known at concentrations \(c_1, c_2, \ldots, c_N\), then for any intermediate concentration \(c\) we compute

\[
A(c,\lambda) \approx A(c_i,\lambda) + 
\frac{c - c_i}{c_{i+1} - c_i}\,\big(A(c_{i+1},\lambda) - A(c_i,\lambda)\big),
\]

for the interval \(c_i \leq c \leq c_{i+1}\). This linear approach provides stability, avoids overfitting, and is efficient enough to be implemented on resource-limited devices such as smartphones.  

Once interpolated spectra are obtained, they are converted into perceivable colors. Each spectrum \(A(\lambda)\) is projected into the CIE color space using the standard color-matching functions \(\bar{x}(\lambda), \bar{y}(\lambda), \bar{z}(\lambda)\). The tristimulus values are computed as

\[
X = \int A(\lambda)\,\bar{x}(\lambda)\,d\lambda, \quad
Y = \int A(\lambda)\,\bar{y}(\lambda)\,d\lambda, \quad
Z = \int A(\lambda)\,\bar{z}(\lambda)\,d\lambda,
\]

which are then normalized and converted into RGB coordinates. This produces a continuous mapping from arsenic concentration to visible color, forming the basis of the sensor’s readout.  

In practice, this workflow is integrated into a phone camera application. A camera records the assay color both before and after exposure to a water sample. The measured RGB values are then matched against the interpolated spectrum-to-color calibration curve described above. This allows the application to estimate the underlying absorbance change and, from that, infer the arsenic concentration. In this way, the phone itself acts as a low-cost spectrometer, enabling field-ready arsenic detection without the need for specialized laboratory equipment.  
