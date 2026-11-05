"""Package this checkpoint without modifying or recomputing numerical fields."""
import hashlib,json,shutil,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[1];out=R/'output/pdf';out.mkdir(parents=True,exist_ok=True)
name='nonlinear_solitary_wave_report'
shutil.copy2(R/f'report/{name}.pdf',out/f'{name}.pdf')
report=[R/f'report/{name}.tex',R/'report/nonlinear_tables.py',*sorted((R/'report/nonlinear_figures').glob('*.pdf')),*sorted((R/'report/nonlinear_tables').glob('*.tex'))]
with zipfile.ZipFile(out/f'{name}_latex.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in report:z.write(p,p.relative_to(R))
 z.writestr('README.txt','Portable LaTeX report: compile from this directory with\n  tectonic report/nonlinear_solitary_wave_report.tex\nThe existing tables and figures are sufficient; no simulations are required.\nA LaTeX compiler with the packages/fonts used in the preamble is required.\nTo regenerate the figures or tables, use the companion solver and numerical data.\n')
raw=list(sorted((R/'results/nonlinear').glob('*.npz')))
manifest={'date':'2026-09-14','raw_field_archives':[{'path':str(p.relative_to(R)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in raw], 'note':'Large evolution NPZ archives are retained in the workspace, omitted from the compact solver ZIP, and reproducible with the included commands. Profile NPZ files are included.'}
(R/'results/nonlinear/raw_field_manifest.json').write_text(json.dumps(manifest,indent=2))
files=[*sorted((R/'nonlinear').glob('*.py')),R/'nonlinear/README.md',R/'nonlinear/requirements.txt',*report,*sorted((R/'results/nonlinear').glob('*.json')),*sorted((R/'results/nonlinear').glob('profile*.npz'))]
with zipfile.ZipFile(out/'nonlinear_solver_and_results.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in files:z.write(p,p.relative_to(R))
 z.writestr('README.txt','Read nonlinear/README.md for equations, scope, commands, output schema, and limitations.\nReport source, saved tables and figure PDFs are included.\nLarge evolution field arrays and the supplied third-party manuscript are excluded.\nAll numerical JSON histories and the computed profile arrays are included.\nRun the complete experiment sequence before rerunning nonlinear.analyze.\n')
for p in [out/f'{name}.pdf',out/f'{name}_latex.zip',out/'nonlinear_solver_and_results.zip']:print(p.name,p.stat().st_size)
