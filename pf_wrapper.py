from peptide_forest.peptide_forest import PeptideForest

def train_run(config, output_path, input_df):
    pf = PeptideForest(
        config_path=config,
        output=output_path,
    )
    pf.input_df = input_df
    pf.fit()
    pf.get_results()
    pf.write_output()


def consolidate_evals(config, output_path, trained_df):
    pf = PeptideForest(
        config_path=config,
        output=output_path,
    )
    pf.trained_df = trained_df
    pf.get_results()
    pf.write_output()