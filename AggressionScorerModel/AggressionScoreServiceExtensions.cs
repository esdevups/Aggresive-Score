using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.ML;

namespace AggressionScorerModel
{
   public static class AggressionScoreServiceExtensions
   {
       private static readonly string _modelFile =
           Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Model", "AggressionScoreRetrainedModel.zip");
        public static void AddAggressionScorePredictionEnginPool(this IServiceCollection services)
        {
            services.AddPredictionEnginePool<ModelInput, ModelOutput>()
                .FromFile(filePath: _modelFile, watchForChanges: true);
        }
    }
}
