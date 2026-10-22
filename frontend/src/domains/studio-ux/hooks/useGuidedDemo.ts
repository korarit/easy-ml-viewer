import { useCallback, useState } from 'react';
import { useDatasetStore } from '../../dataset/store/useDatasetStore';
import { useModelStore } from '../../model_engine/store/useModelStore';
import { useMappingStore } from '../../mapping/store/useMappingStore';
import { useTrajectoryQueueStore } from '../../trajectory_queue/store/useTrajectoryQueueStore';
import { useSystemStore } from '../../../core/store/useSystemStore';

export interface StoryDemo {
  key: string;
  question: string;
  dataset: string;
  model: string;
  outcome: string;
  lookFor: string;
  frames: string;
}

/**
 * One-click stories: each frames a real bundled dataset+model pair as the
 * question it answers, so a first-time user understands the studio in
 * ~30 seconds without touching Steps 1–4 by hand.
 */
export const STORY_DEMOS: StoryDemo[] = [
  {
    key: 'iris-knn',
    question: 'Which measurements decide the flower\u2019s species?',
    dataset: 'sample_iris_knn',
    model: 'sample_iris_knn',
    outcome: 'k-NN votes between 3 species on petal + sepal size',
    lookFor: 'Press play, then open the Why tab — petal length dominates almost every frame.',
    frames: '100 flowers',
  },
  {
    key: 'credit-logreg',
    question: 'Why was this loan application rejected?',
    dataset: 'sample_credit_sgd_log',
    model: 'sample_credit_sgd_log',
    outcome: 'A linear classifier scores credit-default risk',
    lookFor: 'Scrub to a rejected case, read Why, then flip it in What-if with the smallest income change.',
    frames: '100 applications',
  },
  {
    key: 'housing-ridge',
    question: 'What pushes house prices up?',
    dataset: 'sample_housing_ridge',
    model: 'sample_housing_ridge',
    outcome: 'Ridge regression predicts price from housing features',
    lookFor: 'Watch the 3D flight, then check Why — one or two features carry most of the price.',
    frames: '100 houses',
  },
];

/**
 * Loads dataset + model, auto-matches columns, commits the mapping, fills
 * the queue with the first 100 rows and jumps straight to the 3D studio.
 * Throws with a human-readable message; callers surface it and stay put.
 */
export function useGuidedDemo() {
  const [busyKey, setBusyKey] = useState<string | null>(null);
  const [stage, setStage] = useState<string | null>(null);

  const loadStory = useCallback(async (story: StoryDemo) => {
    setBusyKey(story.key);
    try {
      setStage('Loading data…');
      await useDatasetStore.getState().loadSampleDataset(story.dataset);
      const dataset = useDatasetStore.getState().currentDataset;
      if (!dataset) throw new Error('The sample data did not load — is the backend online?');

      setStage('Loading model…');
      await useModelStore.getState().loadSampleModel(story.model);
      const model = useModelStore.getState().currentModel;
      if (!model) throw new Error('The sample model did not load — is the backend online?');

      setStage('Matching columns…');
      await useMappingStore.getState().requestAutoSuggest(dataset.dataset_id, model.model_id);
      const mappingError = useMappingStore.getState().mappingError;
      if (mappingError) throw new Error(mappingError);

      // Auto-pick the leftover non-feature column as the target (actuals) so
      // match badges + the evaluation lens work in the demo (e.g. species).
      const afterSuggest = useMappingStore.getState();
      if (!afterSuggest.targetColumnName) {
        const used = new Set(
          afterSuggest.items
            .map((m) => m.dataset_column_name)
            .filter((c): c is string => !!c),
        );
        const leftover = (dataset.columns ?? [])
          .map((c) => c.name)
          .find((n) => !used.has(n));
        if (leftover) await useMappingStore.getState().setTargetColumn(leftover);
      }

      const ok = await useMappingStore.getState().commitMapping();
      if (!ok) {
        const commitError = useMappingStore.getState().mappingError;
        throw new Error(
          commitError ?? 'Columns did not match cleanly — open Step 3 to fix the highlighted rows.',
        );
      }

      const mapping = useMappingStore.getState();
      const mappedColumns = mapping.items
        .map((m) => m.dataset_column_name)
        .filter((c): c is string => !!c);

      setStage('Queuing frames…');
      await useTrajectoryQueueStore
        .getState()
        .initialize(dataset.dataset_id, mappedColumns, mapping.targetColumnName);
      const queued = useTrajectoryQueueStore.getState().selectedRowIndices.length;
      if (queued === 0) throw new Error('No rows made it into the queue — try the story again.');

      useSystemStore.getState().unlockStep(5);
    } finally {
      setBusyKey(null);
      setStage(null);
    }
  }, []);

  return { busyKey, stage, loadStory };
}
