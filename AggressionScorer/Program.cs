using System;
using System.IO;
using AggressionScorerModel;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers;

namespace AggressionScorer
{
    class Program
    {
        static void Main(string[] args)
        {
            
            Console.WriteLine("Aggression scorer model builder started!");

            var mlContext = new MLContext(0);

            //Load data 

            var createInputFile = @"Data\preparedInput.tsv";
            DataPreparer.CreatePreparedDataFile(createInputFile, true);

            IDataView traninDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                path: createInputFile,
                hasHeader: true,
                separatorChar: '\t',
                allowQuoting: true
            );

            var inputDataSplit = mlContext.Data.TrainTestSplit(traninDataView, testFraction: .2, seed: 0);


            //Build pipeline
            var inputDataPreparer = mlContext
                .Transforms
                .Text
                .FeaturizeText("Features", "Comment")
                .Append(mlContext.Transforms.NormalizeMeanVariance("Features"))
                .AppendCacheCheckpoint(mlContext)
                .Fit(inputDataSplit.TrainSet);

            var trainer = mlContext
                .BinaryClassification
                .Trainers
                .LbfgsLogisticRegression();

            
      
            //Fit the model
            Console.WriteLine("Start Training");
            var transformedData = inputDataPreparer.Transform(inputDataSplit.TrainSet);
            ITransformer model = trainer.Fit(transformedData);


            //Test the model
            EvaluateModel(mlContext, model, inputDataPreparer.Transform(inputDataSplit.TestSet));

            //Save the model
            if (!Directory.Exists("Model"))
            {
                Directory.CreateDirectory("Model");
            }
            var modelFile = @"Model\\AggressionScoreModel.zip";
            mlContext.Model.Save(model,traninDataView.Schema,modelFile);

            Console.WriteLine("Done !");

            var dataPreparePipelineFile = @"Model\\dataPreparePipeline.zip";
            mlContext.Model.Save(inputDataPreparer, traninDataView.Schema, dataPreparePipelineFile);

            var retrainedModel = RetrainModel(modelFile, dataPreparePipelineFile);

            var completeRetrainedPipeline = inputDataPreparer.Append(retrainedModel);

            Console.WriteLine("Saving Retrain Model");

            string retrainModelFile = @"Model\\AggressionScoreRetrainedModel.zip";
            mlContext.Model.Save(completeRetrainedPipeline,traninDataView.Schema,retrainModelFile);

            Console.WriteLine("The model is saved to {0}",retrainModelFile);

            EvaluateModel(mlContext,completeRetrainedPipeline,inputDataSplit.TestSet);
        }

        private static ITransformer RetrainModel(string modelFile, string dataPreparePipelineFile)
        {
            MLContext mlContext=new MLContext(0);

            ITransformer pretrainedModel = mlContext.Model.Load(modelFile, out _);

            var pretrainedModelParameters = ((ISingleFeaturePredictionTransformer
                    <CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>)
                pretrainedModel).Model.SubModel;

            var dataFile = @"Data\preparedInput.tsv";
            DataPreparer.CreatePreparedDataFile(dataFile,false);

            IDataView traninDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                path: dataFile,
                hasHeader: true,
                separatorChar: '\t',
                allowQuoting: true
            );

            ITransformer dataPrepPipeLine = mlContext.Model.Load(dataPreparePipelineFile, out _);

            var newData = dataPrepPipeLine.Transform(traninDataView);

            var retrainModel =
                mlContext.BinaryClassification.Trainers
                    .LbfgsLogisticRegression().Fit(newData, pretrainedModelParameters);

            return retrainModel;
        }

        private static void EvaluateModel(MLContext mlContext, ITransformer trainedData, IDataView testData)
        {
            Console.WriteLine();
            Console.WriteLine("-- Evaluating Binary Classification Model --");
            Console.WriteLine();

            var predictedData = trainedData.Transform(testData);
            var metrics = mlContext.BinaryClassification.Evaluate(predictedData);
            Console.WriteLine($"Accuracy : {metrics.Accuracy:0.###}");

            Console.WriteLine("Confusion Matrix");
            Console.WriteLine();
            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
            Console.WriteLine();

        }
    }
}
