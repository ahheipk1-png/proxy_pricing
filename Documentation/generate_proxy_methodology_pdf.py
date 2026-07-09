from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "proxy_pricing_methodology.pdf"


def page_footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D5D9DE"))
    canvas.line(0.65 * inch, 0.52 * inch, 7.85 * inch, 0.52 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#606872"))
    canvas.drawString(0.68 * inch, 0.34 * inch, "Monte Carlo Proxy Pricing Methodology")
    canvas.drawRightString(7.82 * inch, 0.34 * inch, f"Page {document.page}")
    canvas.restoreState()


styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="TitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=25,
        leading=30,
        textColor=colors.HexColor("#17324D"),
        alignment=TA_CENTER,
        spaceAfter=18,
    )
)
styles.add(
    ParagraphStyle(
        name="Subtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=17,
        textColor=colors.HexColor("#52606D"),
        alignment=TA_CENTER,
        spaceAfter=14,
    )
)
styles.add(
    ParagraphStyle(
        name="H1Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#17324D"),
        spaceBefore=4,
        spaceAfter=10,
    )
)
styles.add(
    ParagraphStyle(
        name="H2Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#24557A"),
        spaceBefore=10,
        spaceAfter=6,
    )
)
styles.add(
    ParagraphStyle(
        name="BodyCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.2,
        leading=13.2,
        textColor=colors.HexColor("#20262D"),
        spaceAfter=6,
    )
)
styles.add(
    ParagraphStyle(
        name="SmallCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7.8,
        leading=10.5,
        textColor=colors.HexColor("#4A5560"),
        spaceAfter=4,
    )
)
styles.add(
    ParagraphStyle(
        name="CodeCustom",
        fontName="Courier",
        fontSize=7.7,
        leading=10.4,
        leftIndent=8,
        rightIndent=8,
        borderColor=colors.HexColor("#D9E2EA"),
        borderWidth=0.6,
        borderPadding=7,
        backColor=colors.HexColor("#F6F8FA"),
        spaceBefore=4,
        spaceAfter=8,
    )
)
styles.add(
    ParagraphStyle(
        name="Callout",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=9.2,
        leading=13,
        textColor=colors.HexColor("#17324D"),
        borderColor=colors.HexColor("#5AA6A6"),
        borderWidth=1,
        borderPadding=8,
        backColor=colors.HexColor("#EDF7F6"),
        spaceBefore=6,
        spaceAfter=10,
    )
)


def p(text, style="BodyCustom"):
    return Paragraph(text, styles[style])


def h1(text):
    return Paragraph(text, styles["H1Custom"])


def h2(text):
    return Paragraph(text, styles["H2Custom"])


def eq(text):
    return Preformatted(text.strip(), styles["CodeCustom"])


def table(data, widths=None, header=True, font_size=7.6):
    item = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDEAF3")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17324D")),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2.5),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C7D0D9")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for row in range(1, len(data)):
        if row % 2 == 0:
            commands.append(
                ("BACKGROUND", (0, row), (-1, row), colors.HexColor("#F7F9FB"))
            )
    item.setStyle(TableStyle(commands))
    return item


def bullet(text):
    return p(f"- {text}")


def add_plot(story, relative_path, caption, max_width=7.0 * inch, max_height=4.8 * inch):
    path = ROOT / relative_path
    if not path.exists():
        return
    image = Image(str(path))
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    story.append(
        KeepTogether(
            [
                Spacer(1, 6),
                image,
                p(caption, "SmallCustom"),
            ]
        )
    )


story = []

# Title
story.extend(
    [
        Spacer(1, 0.75 * inch),
        Paragraph("Monte Carlo Proxy Pricing", styles["TitleCustom"]),
        Paragraph(
            "A step-by-step methodology for European, American, Asian, barrier, cliquet, "
            "SLV cliquet, and three-underlying basket cliquet instruments",
            styles["Subtitle"],
        ),
        Spacer(1, 0.25 * inch),
        p(
            "This guide derives the transformations, estimators, regression bases, "
            "interpolants, variance-reduction methods, dynamic-programming equations, "
            "and validation metrics used in the proxy-pricing experiments.",
            "Callout",
        ),
        Spacer(1, 0.25 * inch),
        table(
            [
                ["Instrument", "Effective state", "Current preferred fitter"],
                ["European call", "1D d1-like moneyness", "PCHIP default; Bernstein optimized"],
                ["American put", "Spot and exercise time", "PCHIP continuation recursion"],
                ["Arithmetic Asian", "1D adjusted moneyness after reduction", "PCHIP default; Akima optimized"],
                ["Barrier call", "Spot plus alive/hit flag", "PCHIP + Brownian bridge"],
                ["GBM cliquet", "1D accrued clipped return", "Bounded Chebyshev degree 19"],
                ["Single-name SLV cliquet", "Accrued, spot, variance", "Local/spectral hybrid"],
                ["3-asset SLV basket cliquet", "Accrued, 3 spots, 3 variances", "Sparse baseline; enrichment needed"],
            ],
            widths=[1.45 * inch, 2.55 * inch, 2.85 * inch],
        ),
        Spacer(1, 0.25 * inch),
        p("Research implementation: proxy_pricing repository", "Subtitle"),
        PageBreak(),
    ]
)

# Contents and scope
story.extend(
    [
        h1("1. Scope and guiding principle"),
        p(
            "A pricing proxy is a fast approximation V_hat(x,t) to an expensive "
            "risk-neutral value V(x,t). The central difficulty is not regression alone. "
            "It is choosing a Markov state, transforming the payoff geometry, generating "
            "unbiased low-noise labels, fitting without violating known bounds, and "
            "testing in the tails."
        ),
        p(
            "The experiments support one broad rule: use product structure to reduce "
            "dimension first, then choose the simplest smoother compatible with the "
            "remaining geometry. A one-dimensional curve and a seven-dimensional SLV "
            "surface should not be forced into the same numerical representation."
        ),
        h2("Contents"),
        table(
            [
                ["Part", "Topic"],
                ["I", "Risk-neutral valuation, Monte Carlo labels, and variance reduction"],
                ["II", "Target transforms, Chebyshev/Bernstein bases, PCHIP, Akima, Bezier"],
                ["III", "European and Asian one-dimensional reductions"],
                ["IV", "American optimal stopping and barrier options"],
                ["V", "Single-name cliquet and SLV extensions"],
                ["VI", "Three-underlying SLV basket cliquet"],
                ["VII", "Generic high-dimensional workflow and validation checklist"],
            ],
            widths=[0.7 * inch, 6.1 * inch],
        ),
        h2("What the reported error means"),
        p(
            "Unless stated otherwise, relative error is abs(proxy-benchmark) divided by "
            "max(abs(benchmark), 0.01). The 0.01 floor prevents an economically tiny "
            "absolute discrepancy near zero from dominating a percentage statistic. "
            "Signed and absolute errors are retained separately."
        ),
        PageBreak(),
    ]
)

# Foundation
story.extend(
    [
        h1("2. Risk-neutral valuation foundation"),
        h2("2.1 Conditional expectation"),
        p(
            "Under a risk-neutral measure Q, with short rate r and state X_t, a European-"
            "style discounted payoff H has value"
        ),
        eq(
            """
V(t,x) = E_Q[ exp(-Integral_t^T r_u du) H(X_T) | X_t = x ].
"""
        ),
        p(
            "Path-dependent contracts become Markov after augmenting X_t with sufficient "
            "path statistics: a running sum for an Asian, accrued clipped return for a "
            "cliquet, and current variances for SLV."
        ),
        h2("2.2 GBM transition"),
        p("For constant parameters under Q:"),
        eq(
            """
dS_t / S_t = (r-q) dt + sigma dW_t.

Integrating d log(S_t) gives
log(S_T/S_t) = (r-q-sigma^2/2) tau + sigma sqrt(tau) Z,
Z ~ N(0,1).

Therefore
S_T = S_t exp((r-q-sigma^2/2) tau + sigma sqrt(tau) Z).
"""
        ),
        p(
            "This exact transition removes time-discretization error for GBM European, "
            "Asian-fixing, and GBM cliquet simulations."
        ),
        h2("2.3 SLV transition"),
        eq(
            """
dS_i/S_i = (r-q_i) dt + L_i(S_i) sqrt(v_i) dW_i^S
dv_i     = kappa_i(theta_i-v_i) dt + xi_i sqrt(v_i) dW_i^v.
"""
        ),
        p(
            "The experiments use full-truncation Euler. With v_plus=max(v,0):"
        ),
        eq(
            """
S_next = S exp((r-q-0.5 L(S)^2 v_plus) dt
               + L(S) sqrt(v_plus dt) Z_S)

v_next = max(v + kappa(theta-v_plus) dt
               + xi sqrt(v_plus dt) Z_v, 0).
"""
        ),
        p(
            "This preserves positivity numerically but introduces discretization bias. "
            "Proxy error is measured against a benchmark using the same grid, so Euler "
            "bias is outside the reported proxy metric."
        ),
        PageBreak(),
    ]
)

# MC and variance reduction
story.extend(
    [
        h1("3. Monte Carlo labels and variance reduction"),
        h2("3.1 Plain estimator and uncertainty"),
        eq(
            """
Y_j = discounted payoff on path j
V_hat_MC = (1/N) Sum_j Y_j
SE(V_hat_MC) = sample_std(Y) / sqrt(N).
"""
        ),
        p(
            "The standard error falls only as N^(-1/2). Ten times smaller noise requires "
            "one hundred times as many paths, so structural variance reduction is more "
            "valuable than path count alone."
        ),
        h2("3.2 Antithetic sampling"),
        eq(
            """
For Z ~ N(0,I), both Z and -Z have the same law.
Use pair average: Y_pair = (Y(Z)+Y(-Z))/2.

Var(Y_pair) = 0.5 Var(Y) + 0.5 Cov(Y(Z),Y(-Z)).
"""
        ),
        p(
            "For monotone payoffs the covariance is often negative, reducing variance "
            "without changing the expectation."
        ),
        h2("3.3 Likelihood-ratio importance sampling"),
        p(
            "Suppose the standard normal is sampled from q=N(mu,I) instead of "
            "p=N(0,I). The density ratio follows directly by completing the square:"
        ),
        eq(
            """
p(z) / q(z)
= exp(-z'z/2 + (z-mu)'(z-mu)/2)
= exp(-mu'z + mu'mu/2).

E_p[H(Z)] = E_q[ H(Z) p(Z)/q(Z) ].
"""
        ),
        p(
            "For multiple independent steps, log likelihoods add. The European and Asian "
            "experiments shift normals toward the payoff boundary. SLV cliquets use a "
            "defensive mixture: half unshifted and half shifted paths."
        ),
        eq(
            """
q_mix(z) = 0.5 p(z) + 0.5 p(z-mu)

p(z)/q_mix(z)
= 1 / [0.5 + 0.5 exp(mu'z - mu'mu/2)].
"""
        ),
        p(
            "The unshifted component protects central states; the shifted component "
            "stabilizes rare positive payoffs near a global floor."
        ),
        PageBreak(),
    ]
)

story.extend(
    [
        h1("4. Control variates"),
        p(
            "Let Y be the target payoff and C a correlated control with known expectation "
            "c. Define Y_beta=Y-beta(C-c). It remains unbiased because E[C-c]=0."
        ),
        eq(
            """
Var(Y_beta) = Var(Y) + beta^2 Var(C) - 2 beta Cov(Y,C).

Differentiate with respect to beta:
2 beta Var(C) - 2 Cov(Y,C) = 0.

beta_star = Cov(Y,C) / Var(C).
"""
        ),
        p(
            "The Asian benchmark uses the exact discrete geometric Asian value as the "
            "control expectation. The GBM cliquet uses the future clipped-return sum, "
            "whose first moment is available from truncated lognormal moments."
        ),
        h2("4.1 Truncated lognormal moment identity"),
        p("If X~N(mu,s^2), then for any real a and threshold b:"),
        eq(
            """
E[ exp(aX) 1_{X <= b} ]
= exp(a mu + 0.5 a^2 s^2)
  Phi((b-mu-a s^2)/s).
"""
        ),
        p(
            "Derivation: multiply the normal density by exp(aX), complete the square, "
            "and recognize a shifted normal density. Differences of this expression over "
            "two thresholds give interval moments. These moments produce exact means and "
            "variances for clip(exp(X)-1, local_floor, local_cap)."
        ),
        h2("4.2 Benchmark independence"),
        p(
            "Training and benchmark random streams are independent. A proxy should not "
            "be evaluated against the same paths used to train it; common random numbers "
            "may make a poor proxy appear artificially accurate."
        ),
        PageBreak(),
    ]
)

# Basis and transforms
story.extend(
    [
        h1("5. Proxy targets and regularized bases"),
        h2("5.1 Coordinate scaling"),
        eq(
            """
x_scaled = 2 (x-x_min)/(x_max-x_min) - 1.
"""
        ),
        p(
            "Chebyshev and Bernstein systems are evaluated on compact intervals. Inputs "
            "outside the trained domain are clipped in these experiments; production "
            "systems should define explicit extrapolation policies."
        ),
        h2("5.2 Ridge regression"),
        eq(
            """
min_beta ||A beta - y||_2^2 + lambda ||Gamma beta||_2^2

Normal equation:
(A'A + lambda Gamma'Gamma) beta = A'y.
"""
        ),
        p(
            "The intercept is not penalized. Ridge stabilizes noisy Monte Carlo labels and "
            "ill-conditioned high-order bases."
        ),
        h2("5.3 Chebyshev basis"),
        eq(
            """
T_0(x)=1, T_1(x)=x,
T_(n+1)(x)=2x T_n(x)-T_(n-1)(x).

Proxy: y_hat(x)=Sum_(k=0)^d beta_k T_k(x).
"""
        ),
        p(
            "Chebyshev polynomials are better conditioned on [-1,1] than monomials and "
            "work well for globally smooth value functions."
        ),
        h2("5.4 Bernstein and global Bezier basis"),
        eq(
            """
B_(k,n)(u) = C(n,k) u^k (1-u)^(n-k),  0<=u<=1.
y_hat(u) = Sum_(k=0)^n c_k B_(k,n)(u).
"""
        ),
        p(
            "The Bernstein basis is nonnegative and sums to one. With fitted coefficients "
            "c_k it is a global Bezier curve. It provided the best European 1D result."
        ),
        h2("5.5 Log and bounded-logit targets"),
        eq(
            """
Positive value: y = log(V + epsilon), V_hat=exp(y_hat)-epsilon.

Known bounds L<U:
p=(V-L)/(U-L), y=log(p/(1-p)),
V_hat=L+(U-L)/(1+exp(-y_hat)).
"""
        ),
        p(
            "The log target prioritizes relative accuracy. The logit target guarantees "
            "hard global floor/cap bounds for cliquets."
        ),
        PageBreak(),
    ]
)

# Interpolants
story.extend(
    [
        h1("6. PCHIP, Akima, and piecewise Bezier curves"),
        h2("6.1 Cubic Hermite segment"),
        eq(
            """
t=(x-x_i)/h, h=x_(i+1)-x_i

H(t)=h00(t)y_i + h10(t)h d_i
     + h01(t)y_(i+1) + h11(t)h d_(i+1)

h00=2t^3-3t^2+1, h10=t^3-2t^2+t
h01=-2t^3+3t^2, h11=t^3-t^2.
"""
        ),
        h2("6.2 PCHIP slope rule"),
        p(
            "Let secants delta_i=(y_(i+1)-y_i)/h_i. If adjacent secants have opposite "
            "sign, PCHIP sets the interior derivative to zero. Otherwise it uses a "
            "weighted harmonic mean:"
        ),
        eq(
            """
d_i = (w1+w2)/(w1/delta_(i-1)+w2/delta_i)
w1=2h_i+h_(i-1), w2=h_i+2h_(i-1).
"""
        ),
        p(
            "This suppresses overshoot and preserves monotonicity. It is especially useful "
            "for American continuation curves near an exercise boundary."
        ),
        h2("6.3 Akima slope rule"),
        eq(
            """
d_i = [w_left delta_(i-1) + w_right delta_i]
      / (w_left+w_right)

w_left  = |delta_(i+1)-delta_i|
w_right = |delta_(i-1)-delta_(i-2)|.
"""
        ),
        p(
            "Akima weights the less rapidly changing side more heavily. It is local and "
            "less aggressively monotone than PCHIP. This matched the reduced Asian curve "
            "very well."
        ),
        h2("6.4 Bezier/Hermite equivalence"),
        eq(
            """
For the same endpoint slopes:
P0=y_i
P1=y_i + h d_i/3
P2=y_(i+1) - h d_(i+1)/3
P3=y_(i+1).

B(t)=(1-t)^3 P0 + 3(1-t)^2 t P1
     +3(1-t)t^2 P2 + t^3 P3.
"""
        ),
        p(
            "Expanding B(t) gives the Hermite polynomial exactly. Therefore a piecewise "
            "Bezier curve built from PCHIP slopes is PCHIP in another representation, "
            "not a distinct smoother."
        ),
        PageBreak(),
        h2("6.5 Expanded 99-case smoother study"),
        p(
            "Seventeen distinct one-dimensional estimators were tested over European, "
            "American, Asian, and barrier parameter combinations. Every fitted surface "
            "used a 10M-scenario training budget; MC benchmarks used 500,000 paths per "
            "state. Product families received equal aggregate weight."
        ),
        table(
            [
                ["Method", "Balanced p99", "Max for V>=0.05", "Local overshoot"],
                ["Natural cubic interpolation", "1.799%", "1.163%", "2.01%"],
                ["PCHIP", "1.834%", "1.220%", "0.00%"],
                ["Akima", "1.958%", "1.286%", "0.17%"],
                ["MAKIMA", "1.968%", "1.317%", "0.15%"],
                ["Cubic smoothing spline", "2.098%", "1.440%", "2.49%"],
                ["Linear interpolation", "2.683%", "2.058%", "0.00%"],
            ],
            widths=[2.4 * inch, 1.35 * inch, 1.55 * inch, 1.35 * inch],
        ),
        p(
            "Natural cubic interpolation led average accuracy by only 0.035 percentage "
            "points. PCHIP won 53 of 99 head-to-head p99 cases and introduced no local "
            "overshoot, so PCHIP remains the generic one-feature PFE default."
        ),
        PageBreak(),
    ]
)

# European
story.extend(
    [
        h1("7. European option proxy"),
        h2("7.1 State and benchmark"),
        p(
            "For constant-parameter GBM, the state at a fixed time is spot alone. The "
            "benchmark is the Black-Scholes formula. For a call:"
        ),
        eq(
            """
d1=[log(S/K)+(r-q+sigma^2/2)tau]/(sigma sqrt(tau))
d2=d1-sigma sqrt(tau)
C=S exp(-q tau) Phi(d1)-K exp(-r tau) Phi(d2).
"""
        ),
        h2("7.2 Why d1 is the fitting coordinate"),
        p(
            "d1 combines spot, strike, volatility, carry, and remaining maturity into the "
            "standardized coordinate controlling exercise probability and delta. Sampling "
            "uniformly in d1 is equivalent to deliberate wing coverage in delta space."
        ),
        h2("7.3 Training label"),
        p(
            "For each d1 state, shifted antithetic terminal normals produce an unbiased MC "
            "label. The default target is log(value), originally fitted by degree-7 "
            "Chebyshev ridge."
        ),
        h2("7.4 One-dimensional method comparison"),
        table(
            [
                ["Method", "Worst max error", "Average p99", "Average MAE"],
                ["Global Bernstein/Bezier", "0.536%", "0.208%", "0.003546"],
                ["Chebyshev", "0.854%", "0.326%", "0.004423"],
                ["Akima", "2.540%", "1.127%", "0.007015"],
                ["PCHIP", "2.554%", "1.149%", "0.007098"],
                ["Bezier with PCHIP slopes", "2.554%", "1.149%", "0.007098"],
            ],
            widths=[2.5 * inch, 1.35 * inch, 1.35 * inch, 1.35 * inch],
        ),
        p(
            "Conclusion: the European curve is sufficiently smooth that global regularized "
            "bases average MC noise better than exact local interpolation."
        ),
        p(
            "For implementation consistency, the standalone universal default is now "
            "log-PCHIP in d1. Its independent-seed worst error is 2.637%, versus the "
            "product-specific Bernstein result of 0.536%."
        ),
        PageBreak(),
    ]
)

# Asian derivation
story.extend(
    [
        h1("8. Arithmetic Asian option proxy"),
        h2("8.1 Raw Markov state"),
        p(
            "At fixing index j, store current spot S and running sum A of fixings strictly "
            "before today. Let N be total fixings and m=N-j-1 future fixings."
        ),
        eq(
            """
Payoff = [ (A + S + Sum_(k=1)^m S_(j+k))/N - K ]_+.
"""
        ),
        h2("8.2 Adjusted-strike derivation"),
        eq(
            """
Payoff
= (m/N) [ (1/m) Sum_(k=1)^m S_(j+k) - K_adj ]_+

K_adj = (N K - A - S)/m.
"""
        ),
        p(
            "This is an algebraic identity. It rotates the diagonal payoff boundary in "
            "(S,A) into a strike-like scalar."
        ),
        h2("8.3 GBM homogeneity reduction"),
        p(
            "Future GBM spots satisfy S_(j+k)=S G_k, where the joint growth vector G does "
            "not depend on S. Therefore:"
        ),
        eq(
            """
V(S,A,j)
= (m/N) E[ (S average(G)-K_adj)_+ ]
= S (m/N) E[ (average(G)-K_adj/S)_+ ].
"""
        ),
        p(
            "At a fixed date the normalized value V/S is a one-dimensional function of "
            "K_adj/S, represented by an adjusted d1-like coordinate. This is why a generic "
            "2D fit is unnecessary for this GBM product."
        ),
        h2("8.4 Exact linear wing"),
        p(
            "If K_adj<=0 for a call, every future arithmetic average is positive relative "
            "to the adjusted strike, so the positive-part operator can be removed. The "
            "value is then the discounted expected average minus strike and is computed "
            "analytically."
        ),
        PageBreak(),
    ]
)

story.extend(
    [
        h1("9. Asian benchmark and fitting results"),
        h2("9.1 Geometric Asian control"),
        p(
            "The log of the discrete future geometric average is a weighted sum of normal "
            "increments, hence normal. If weights are w_l=(m-l+1)/m:"
        ),
        eq(
            """
log(G_geo) = log(S)
 + (r-q-sigma^2/2) dt (m+1)/2
 + sigma sqrt(dt) Sum_l w_l Z_l.

Var(log(G_geo)) = sigma^2 dt Sum_l w_l^2.
"""
        ),
        p(
            "Thus the geometric Asian call has a Black-Scholes-like exact value and is a "
            "highly correlated control for the arithmetic payoff."
        ),
        h2("9.2 Hybrid target"),
        p(
            "Fit log(V/S) in the OTM and moderate region. In the deep ITM region fit "
            "log((V-linear_baseline)/S), then add the exact baseline. This prevents a "
            "large intrinsic component from hiding time-value error."
        ),
        h2("9.3 1D fitter comparison"),
        table(
            [
                ["Method", "Worst max error", "Average p99", "Average MAE"],
                ["Akima", "0.186%", "0.057%", "0.000178"],
                ["Chebyshev", "3.286%", "1.721%", "0.013523"],
                ["PCHIP", "3.453%", "0.936%", "0.000273"],
                ["Global Bernstein/Bezier", "6.854%", "3.303%", "0.017907"],
                ["Bezier with PCHIP slopes", "3.453%", "0.936%", "0.000273"],
            ],
            widths=[2.5 * inch, 1.35 * inch, 1.35 * inch, 1.35 * inch],
        ),
        p(
            "Akima is the preferred 1D fitter for this reduced Asian experiment. PCHIP is "
            "usable, but its monotonicity limiter can flatten local derivatives when MC "
            "labels contain small slope reversals. A second independent-seed run confirmed "
            "Akima at 0.144% worst error, versus 2.577% for Chebyshev and 1.643% for PCHIP."
        ),
        p(
            "The standalone universal default is nevertheless PCHIP. With its independent "
            "default seed it achieved 3.331% worst error while sharing the same fitting "
            "engine as European and American options."
        ),
    ]
)
add_plot(
    story,
    "AsianOptExperiment/results/plots/asian_day_06_hybrid.png",
    "Asian adjusted-moneyness proxy diagnostic from the broader experiment.",
)
story.append(PageBreak())

# American
story.extend(
    [
        h1("10. American put: optimal stopping"),
        h2("10.1 Snell envelope and Bellman recursion"),
        p(
            "For exercise dates t_k, intrinsic payoff g(S)=(K-S)_+. The value is the "
            "smallest supermartingale dominating g, equivalently the Snell envelope:"
        ),
        eq(
            """
V_M(S)=g(S)
C_k(S)=exp(-r dt) E[ V_(k+1)(S_next) | S_k=S ]
V_k(S)=max(g(S), C_k(S)).
"""
        ),
        p(
            "The max creates a moving derivative kink at the exercise boundary. A global "
            "polynomial fitted recursively can turn a small local continuation error into "
            "a systematic bias at every earlier date."
        ),
        h2("10.2 MC dynamic programming"),
        p(
            "At each of 100 exercise dates, the implementation uses 121 log-spaced state "
            "nodes, antithetic one-step GBM transitions, and about 10 million transitions "
            "over the complete backward pass. It fits C_k, then imposes max(g,C_k) exactly."
        ),
        h2("10.3 Why PCHIP wins here"),
        p(
            "PCHIP is local, shape preserving, and does not ring around the moving boundary. "
            "The broad experiment achieved 2.249% worst max error; an independent seed "
            "achieved 5.289%. Global log-Chebyshev fits were unstable because exponentiated "
            "errors recursively compounded."
        ),
        p(
            "Akima was also tested through the complete 100-date recursion. It achieved "
            "3.225% worst max error, 1.675% average p99, and 0.002033 average MAE, versus "
            "2.249%, 0.947%, and 0.001479 for PCHIP. Akima is viable, but PCHIP's "
            "monotonicity-preserving slopes are safer near the exercise boundary."
        ),
        h2("10.4 Independent benchmark"),
        p(
            "A projected implicit finite-difference solver uses 4,000 time steps and 2,000 "
            "spot steps. Projected SOR enforces V>=g. At S=100, t=0, the finite-difference "
            "value is about 6.6597 versus 6.6605 from a separate 4,000-step CRR tree."
        ),
    ]
)
add_plot(
    story,
    "AmericanOptExperiment/results/plots/american_step_040_pchip_spline.png",
    "American put value, signed error, and signed relative error at time 0.4.",
)
story.append(PageBreak())

# Barrier options
story.extend(
    [
        h1("10A. Single and double barrier options"),
        h2("10A.1 State and variants"),
        p(
            "The experiment prices zero-rebate down-and-out, up-and-out, and "
            "double-knock-out calls with either monthly discrete or continuous "
            "monitoring. Conditional on no historical hit, spot is the only fitted "
            "feature. A live/hit indicator must be added once monitoring history exists."
        ),
        p(
            "Zero-rebate knock-ins require no second MC training run. Under the same "
            "monitoring convention, in/out parity gives V_in = V_vanilla - V_out."
        ),
        h2("10A.2 Exact Brownian-bridge segment correction"),
        p(
            "For log endpoints x and y above lower log barrier b over a GBM step with "
            "variance q=sigma^2 dt, reflection gives the conditional survival probability"
        ),
        eq(
            """
P(min bridge > b | x,y)
  = 1 - exp[-2 (x-b)(y-b)/q],  x>b, y>b.
"""
        ),
        p(
            "The upper-barrier expression replaces the distances by b-x and b-y. "
            "Multiplying segment survival probabilities and weighting the payoff "
            "integrates out crossing indicators, reducing variance. For a double "
            "barrier, the absorbing interval transition density is evaluated by a "
            "method-of-images series and divided by the free transition density."
        ),
        h2("10A.3 Sampling and results"),
        p(
            "Log-Chebyshev spot nodes cover the alive domain. Common random numbers make "
            "label noise smooth across spot, antithetics reduce path noise, and shifted "
            "interior validation nodes prevent training-grid leakage."
        ),
        table(
            [
                ["Variant", "PCHIP raw worst", "Worst for V>=0.05"],
                ["Down-out discrete", "2.253%", "2.087%"],
                ["Down-out continuous", "2.531%", "1.762%"],
                ["Up-out discrete", "9.096%", "4.853%"],
                ["Up-out continuous", "8.808%", "5.074%"],
                ["Double-out discrete", "8.387%", "4.999%"],
                ["Double-out continuous", "8.148%", "4.916%"],
            ],
            widths=[2.5 * inch, 1.8 * inch, 2.0 * inch],
        ),
        p(
            "The 8-9% raw maxima are one- or two-cent option values. Meaningful-value "
            "errors remain around 2% for down barriers and about 5% for up/double "
            "barriers. PCHIP remains the default because the broader 99-case study found "
            "near-best accuracy with zero local overshoot."
        ),
    ]
)
add_plot(
    story,
    "BarrierOptExperiment/results/plots/double_out_continuous_month_06_pchip.png",
    "Continuous double-knock-out value, signed error, and signed relative error.",
)
story.append(PageBreak())

# Cliquet
story.extend(
    [
        h1("11. Single-name GBM cliquet"),
        h2("11.1 Payoff"),
        eq(
            """
c_i = clip(S_i/S_(i-1)-1, local_floor, local_cap)
H = notional * clip(Sum_i c_i, global_floor, global_cap).
"""
        ),
        h2("11.2 State reduction theorem"),
        p(
            "Under GBM, each future reset return S_i/S_(i-1)-1 depends only on the new "
            "normal increment and fixed model parameters, not on S_(i-1). Future coupons "
            "are therefore independent of current spot conditional on reset time."
        ),
        eq(
            """
State at reset = accrued = Sum of realized clipped coupons.
V = V(accrued, remaining_periods).
"""
        ),
        p(
            "This is an exact scale-invariance reduction, not an empirical approximation."
        ),
        h2("11.3 Expected-total coordinate"),
        eq(
            """
z = [accrued + m E(c) - midpoint(global bounds)]
    / sqrt(m Var(c)).
"""
        ),
        p(
            "A boundary-enriched grid samples around the estimated floor and cap transitions. "
            "Known flat tails are exact: if even all remaining local caps cannot leave the "
            "global floor, or all local floors force the cap, regression is bypassed."
        ),
        h2("11.4 Result"),
        p(
            "Bounded-logit Chebyshev degree 19 achieved 3.593% worst max error, 0.928% "
            "average p99, and 0.003194 average MAE."
        ),
    ]
)
add_plot(
    story,
    "CliquetOptExperiment/results/plots/cliquet_day_09_logit_z_boundary_d19.png",
    "GBM cliquet proxy diagnostic with three coupons remaining.",
)
story.append(PageBreak())

# SLV cliquet
story.extend(
    [
        h1("12. Single-name SLV cliquet"),
        h2("12.1 Expanded state"),
        p(
            "SLV destroys GBM scale invariance because leverage L(S) depends on spot and "
            "future variance depends on current v. The reset state becomes "
            "(accrued, S, v)."
        ),
        h2("12.2 Structured features"),
        p(
            "A frozen-coefficient one-period coupon distribution supplies conditional mean "
            "and variance. These form an expected-total coordinate z, supplemented by "
            "log(S/S0) and log(v/theta). The target remains bounded logit."
        ),
        h2("12.3 Maturity-adaptive fit"),
        table(
            [
                ["Remaining coupons", "Fitter", "Reason"],
                ["9-12", "Local quadratic", "Broad smooth long-horizon dependence"],
                ["3-6", "Anisotropic Chebyshev", "Sharper payoff transitions"],
                ["0", "Exact payoff", "No continuation uncertainty"],
            ],
            widths=[1.45 * inch, 2.0 * inch, 3.35 * inch],
        ),
        p(
            "Lower-tail mixture importance sampling was necessary for seed stability. The "
            "common-random-number rerun achieved 5.067% worst error. A separate two-design "
            "study rebuilt 10M-path labels and used 500,000 benchmark paths per state on "
            "both designs; the fixed adaptive hybrid achieved 4.795% worst error."
        ),
        p(
            "Anisotropic Chebyshev degrees 13-23, sparse Hermite, local regression, and "
            "Nystrom Matern regression were tested in that study. None beat the existing "
            "time-based local/spectral rule. Common random numbers are retained because "
            "they smooth label noise across state without changing conditional means."
        ),
    ]
)
add_plot(
    story,
    "SLVCliquetOptExperiment/results/plots/slv_cliquet_day_09_adaptive_hybrid.png",
    "Single-name SLV cliquet validation states ranked by benchmark value.",
)
story.append(PageBreak())

# Basket
story.extend(
    [
        h1("13. Three-underlying SLV basket cliquet"),
        h2("13.1 Three coupon definitions"),
        table(
            [
                ["Variant", "Monthly coupon"],
                ["Basket return", "clip(mean(R1,R2,R3), local floor, local cap)"],
                ["Average clipped", "mean(clip(R1),clip(R2),clip(R3))"],
                ["Worst of", "clip(min(R1,R2,R3), local floor, local cap)"],
            ],
            widths=[1.55 * inch, 5.25 * inch],
        ),
        h2("13.2 Seven-dimensional Markov state"),
        eq(
            """
X = (accrued, S1,S2,S3, v1,v2,v3).
"""
        ),
        p(
            "Each asset has its own variance factor and leverage function. Correlated market "
            "drivers create cross-asset dependence; each spot shock also contains its own "
            "negatively correlated variance shock."
        ),
        eq(
            """
Z_i^S = rho_i Z_i^v + sqrt(1-rho_i^2) Z_i^M,
Corr(Z^M)=R_market.
"""
        ),
        p(
            "This construction is automatically positive semidefinite because it is a "
            "linear transformation of independent variance normals and a valid correlated "
            "market-normal vector."
        ),
        h2("13.3 Sampling"),
        p(
            "2,001 low-discrepancy boundary-enriched states cover accrued return, three log "
            "spots, and three log variances. Approximately 10 million scenarios are spread "
            "over each reset-date fit. Each of 31 independent validation states uses "
            "500,000 paths."
        ),
        h2("13.4 Features and bounded target"),
        p(
            "Conditional coupon moments define separate lower and upper standardized "
            "cushions. Keeping both is important: one midpoint z cannot distinguish states "
            "with equal standardized midpoint but different dispersion and distances to "
            "the two global bounds."
        ),
        PageBreak(),
    ]
)

story.extend(
    [
        h1("14. Basket high-dimensional model search"),
        p(
            "The search compared local full-state and summary regressions, sparse anisotropic "
            "Chebyshev terms, Gaussian RBF kernels, and a two-layer tanh neural ensemble. "
            "With averaged state labels, neither RBF nor the small neural ensemble was "
            "uniformly competitive. This does not reject neural networks in general; it "
            "shows that architecture and state coverage matter."
        ),
        h2("14.1 Development ensemble and validation failure"),
        eq(
            """
V_proxy = w(m) V_local + [1-w(m)] V_spectral
w(m) = 0.16 + 0.02 m,
m = remaining coupons.
"""
        ),
        p(
            "For symmetric basket-return and average-clipped coupons, V_local used summary "
            "features. Worst-of used the full state. The blend performed well on the first "
            "development design but did not generalize, so it was retired."
        ),
        table(
            [
                ["Variant", "Development", "Validation 2", "Validation 3", "Worst"],
                ["Basket return", "10.258%", "7.108%", "18.500%", "18.500%"],
                ["Average clipped", "5.624%", "11.566%", "5.745%", "11.566%"],
                ["Worst of", "9.700%", "9.862%", "6.774%", "9.862%"],
            ],
            widths=[1.55 * inch, 1.3 * inch, 1.3 * inch, 1.3 * inch, 1.1 * inch],
        ),
        p(
            "These are fixed sparse-spectral results using 2,001 states. A 5,001-state "
            "design, local models, RBF kernels, small tanh networks, random-feature "
            "networks, and bagging were also tested. None guaranteed 5-8% on the third "
            "untouched design. The next credible step is adaptive state enrichment around "
            "failed neighborhoods, followed by another untouched validation design."
        ),
        h2("14.2 Literature-inspired residual search"),
        p(
            "A clipped-normal moment baseline was used as a low-fidelity model, followed "
            "by direct or residual sparse Hermite, local, Nystrom Matern, and fixed "
            "ensemble corrections. Coupon skewness and floor/cap masses were added as "
            "features. Inverse-variance weighted sparse Chebyshev fits were also tested "
            "on both 2,001 and 5,001 states."
        ),
        p(
            "None improved worst-case relative error on the development and independent "
            "designs. Some dense weighted fits reduced MAE but leaked larger errors into "
            "tail states. The moment baseline did not simplify correlated clipped and "
            "worst-of SLV coupons enough for residual learning to help."
        ),
    ]
)
add_plot(
    story,
    "BasketCliquetOptExperiment/results/plots/basket_return_month_09_sparse_chebyshev.png",
    "Development diagnostic for the fixed sparse basket-return proxy with three coupons remaining.",
)
story.append(PageBreak())

# Generic methodology
story.extend(
    [
        h1("15. Generic workflow for a new instrument"),
        h2("Step 1: define the contract exactly"),
        bullet("Observation schedule, exercise rights, local/global bounds, and settlement."),
        bullet("Clarify basket aggregation order: clip-then-average is not average-then-clip."),
        h2("Step 2: identify a sufficient Markov state"),
        bullet("Start from the simulator state and add path statistics required by payoff."),
        bullet("Prove any dimension reduction from scale invariance or payoff algebra."),
        h2("Step 3: nondimensionalize"),
        bullet("Use moneyness, normalized variance, expected-total cushions, and time."),
        bullet("Align transition boundaries across maturities before fitting."),
        h2("Step 4: exploit exact regions"),
        bullet("Terminal payoff, linear ITM wings, hard global floor/cap tails, intrinsic value."),
        h2("Step 5: design state sampling"),
        bullet("Combine broad low-discrepancy coverage with boundary-focused clusters."),
        bullet("Do not spend most paths in flat regions that are already known analytically."),
        h2("Step 6: generate MC labels"),
        bullet("Use antithetic paths by default."),
        bullet("Add controls when exact expectations are available."),
        bullet("Use likelihood-ratio shifts for rare payoff wings."),
        h2("Step 7: choose fitter by effective dimension"),
        table(
            [
                ["Effective dimension", "Default candidates"],
                ["Any genuine 1D price proxy", "PCHIP operational default"],
                ["1D smooth, product optimized", "Natural cubic, Chebyshev, Bernstein"],
                ["1D noisy labels", "Smoothing spline or P-spline after validation"],
                ["2-5D smooth", "Sparse anisotropic basis, local polynomial, small NN"],
                ["6D+", "Smooth NN or sparse/additive model; ensemble after validation"],
            ],
            widths=[1.7 * inch, 5.1 * inch],
        ),
        h2("Step 8: validate independently"),
        bullet("Use new random seeds and a stronger benchmark."),
        bullet("Report value, signed error, relative error, and benchmark standard error."),
        bullet("Slice results by maturity, payoff region, and state-space boundary."),
        bullet("For PFE, run the proxy through outer scenarios and compare exposure quantiles by date."),
        PageBreak(),
    ]
)

story.extend(
    [
        h1("16. Why there is no universal fitter"),
        p(
            "PCHIP works well for the American continuation curve because the geometry is "
            "one-dimensional, monotone, and kinked. It is not directly defined on a general "
            "multidimensional cloud. Tensor-product PCHIP would grow exponentially with "
            "dimension and requires a structured grid."
        ),
        p(
            "Chebyshev and Bernstein bases are excellent global smoothers in one dimension. "
            "In seven dimensions, unrestricted tensor bases are combinatorial. Sparse term "
            "selection helps, but sharp payoff transitions and interactions can still "
            "produce bias."
        ),
        p(
            "Neural networks are a genuinely generic high-dimensional candidate because "
            "parameter count can grow without a tensor grid. Their success requires enough "
            "state diversity, regularization, architecture tuning, and independent seeds. "
            "The small basket network tested here did not beat the structured ensemble."
        ),
        p(
            "A robust universal <i>workflow</i> is more realistic than a universal fitter: "
            "state reduction, payoff-aware features, bounded targets, variance-reduced "
            "labels, dimension-appropriate smoothers, and independent validation."
        ),
        h2("Recommended hierarchy"),
        table(
            [
                ["Question", "Action"],
                ["Can the state be proved 1D?", "Use and compare global ridge, Akima, PCHIP."],
                ["Are hard bounds known?", "Use exact tails and bounded-logit target."],
                ["Is there a dominant baseline?", "Fit residual or time value."],
                ["Do local/global models err oppositely?", "Use a simple cross-validated blend."],
                ["Does dimension exceed sparse-basis capacity?", "Increase state coverage and test smooth NN."],
            ],
            widths=[2.25 * inch, 4.55 * inch],
        ),
        PageBreak(),
    ]
)

# Error and implementation
story.extend(
    [
        h1("17. Error accounting"),
        h2("17.1 Metrics"),
        eq(
            """
signed_error = proxy - benchmark
absolute_error = |signed_error|
relative_error = absolute_error / max(|benchmark|, 0.01)
noise_ratio = absolute_error / max(benchmark_SE, tiny).
"""
        ),
        p(
            "A low noise ratio indicates the discrepancy may be statistically unresolved. "
            "A large noise ratio indicates model/fitting bias rather than benchmark noise."
        ),
        h2("17.2 Error decomposition"),
        eq(
            """
proxy - true continuous-model value
= fitting error
 + training MC error
 + benchmark estimation comparison noise
 + time-discretization bias
 + model/calibration error.
"""
        ),
        p(
            "The reported studies isolate fitting plus training noise against a benchmark "
            "under the same model discretization. They do not claim market calibration "
            "accuracy."
        ),
        h2("17.3 Leakage controls"),
        bullet("Independent training and benchmark random streams."),
        bullet("Fixed validation states not reused as training states."),
        bullet("A final standalone run with an independent seed where provided."),
        bullet("Method-selection results should be confirmed on a second validation design."),
        h2("17.4 Stability checks"),
        bullet("Repeat labels with different seeds."),
        bullet("Double Euler steps for SLV and compare."),
        bullet("Expand spot/variance domains to test extrapolation."),
        bullet("Check price bounds and monotonicity numerically."),
        bullet("For Greeks, validate derivative smoothness separately from price error."),
        PageBreak(),
    ]
)

story.extend(
    [
        h1("18. Reproducibility map"),
        table(
            [
                ["Purpose", "Entry point or folder"],
                ["European default", "EuroMain.py"],
                ["Asian default", "AsianMain.py"],
                ["Barrier research/default", "BarrierOptExperiment/BarrierMain.py"],
                ["GBM cliquet default", "CliquetMain.py"],
                ["American put default", "AmericanMain.py"],
                ["Single-name SLV cliquet", "SLVCliquetMain.py"],
                ["3-asset SLV basket cliquet", "BasketCliquetOptExperiment/BasketCliquetMain.py"],
                ["Expanded 1D comparison", "OneDimensionalFitExperiment/ExpandedSplineStudy.py"],
                ["European research", "EuroOptExperiment/"],
                ["Asian research", "AsianOptExperiment/"],
                ["Cliquet research", "CliquetOptExperiment/"],
                ["American research", "AmericanOptExperiment/"],
                ["SLV research", "SLVCliquetOptExperiment/ and BasketCliquetOptExperiment/"],
            ],
            widths=[2.0 * inch, 4.8 * inch],
            font_size=7.3,
        ),
        h2("Core numerical budgets"),
        table(
            [
                ["Study", "Training", "Benchmark"],
                ["European", "121 states x 25,000 shifted paths", "Black-Scholes closed form"],
                ["Asian", "about 10M scenarios per fitted date", "500K paths/state + geometric control"],
                ["GBM cliquet", "about 10M scenarios per fitted date", "500K paths/state + clipped-sum control"],
                ["American", "10,006,700 one-step transitions", "4,000 x 2,000 projected FD grid"],
                ["Barrier", "10M scenarios per fitted date", "500K paths/state + bridge survival"],
                ["Single SLV cliquet", "about 10M scenarios per fitted date", "500K paths/state"],
                ["Basket SLV cliquet", "about 10M scenarios over 2,001 states/date", "500K paths x 31 states/date"],
            ],
            widths=[1.55 * inch, 2.65 * inch, 2.6 * inch],
            font_size=7.0,
        ),
        h2("Known limitations"),
        bullet("Illustrative SLV leverage functions are not calibrated market surfaces."),
        bullet("SLV uses two Euler steps per monthly period in the current experiments."),
        bullet("Basket 5-8% control failed on untouched designs; adaptive enrichment remains research."),
        bullet("Pointwise price accuracy does not replace outer-scenario PFE quantile validation."),
        bullet("Production extrapolation, Greeks, and calibration loops require separate tests."),
        bullet("The neural basket test is not a definitive neural-network architecture search."),
        PageBreak(),
    ]
)

# Conclusion
story.extend(
    [
        Spacer(1, 0.4 * inch),
        h1("19. Final conclusions"),
        p(
            "The strongest improvement repeatedly came from changing the representation "
            "before changing the regressor. d1 organizes European wings; adjusted strike "
            "reduces the GBM Asian state; expected-total cushions organize cliquet bounds; "
            "the Snell recursion isolates American continuation."
        ),
        p(
            "The expanded one-dimensional test covered 99 European, American, Asian, and "
            "barrier cases with 17 estimators. Natural cubic interpolation led balanced "
            "p99 by only 0.035 percentage points, while PCHIP won 53 of 99 head-to-head "
            "cases and had zero local overshoot."
        ),
        p(
            "When one implementation must cover every genuinely one-dimensional option, "
            "PCHIP is therefore the universal default. Product-specific optimized profiles "
            "remain available when maximum pointwise accuracy matters. For PFE, the final "
            "acceptance metric must be exposure-quantile distortion in actual outer "
            "scenarios, not price RMSE alone."
        ),
        p(
            "For higher-dimensional SLV baskets, sparse payoff-aware regression was the "
            "strongest generic baseline, but three independent state designs showed that "
            "the 10M-path budget did not guarantee the 5-8% target. Robust worst errors "
            "were approximately 18.5%, 11.6%, and 9.9%."
        ),
        p(
            "Moment-residual Hermite, local, Nystrom Matern, weighted spectral, and fixed "
            "ensemble alternatives did not improve the 7D basket tail. This supports "
            "adaptive state enrichment or a larger path-level neural program rather than "
            "further tuning on the same sparse labels."
        ),
        p(
            "The reusable methodology is therefore: derive the state, reduce dimension when "
            "mathematically valid, enforce exact structure, create unbiased low-noise labels, "
            "fit with a dimension-appropriate smoother, and validate independently in the "
            "wings and transition regions.",
            "Callout",
        ),
        Spacer(1, 0.2 * inch),
        p(
            "End of methodology guide.",
            "Subtitle",
        ),
    ]
)


def build():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.62 * inch,
        bottomMargin=0.68 * inch,
        title="Monte Carlo Proxy Pricing Methodology",
        author="Proxy Pricing Research Project",
        subject="European, American, Asian, barrier, cliquet, and basket cliquet proxies",
    )
    document.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    print(OUTPUT)


if __name__ == "__main__":
    build()
