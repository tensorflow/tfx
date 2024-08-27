/*
Copyright 2023 Google LLC. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/
let CONSTANT_VALUES = {
  GLOBALS: {
    LABELS: {
      STALE: 'stale',
      AWAITINGRES: 'stat:awaiting response',
      BUG: 'type:bug',
      BUG_INSTALL: 'type:build/install',
      TYPE_SUPPORT: 'type:support',
      TYPE_OTHER: 'type:others',
      TYPE_DOCS_BUG: 'type:docs-bug',
      TYPE_PERFORMANCE: 'type:performance'
    },
    STATE: {CLOSED: 'closed'},
    TENSORFLOW_CORE_REPO: 'https://github.com/tensorflow/tensorflow/pull',
    PR_TRIGGER_REPO: 'testRep,keras'
  },
  MODULE: {
    CSAT: {
      YES: 'Yes',
      NO: 'No',
      BASE_URL:
          'https://docs.google.com/forms/d/e/1FAIpQLSfaP12TRhd9xSxjXZjcZFNXPG' +
          'k4kc1-qMdv3gc6bEP90vY1ew/viewform?',
      MEDIA_PIPE_BASE_URL:
          'https://docs.google.com/forms/d/e/1FAIpQLScOLT8zeBHummIZFnfr9wqvxYzWD1DAypyvNia5WVIWtFANYg/viewform?',
      SATISFACTION_PARAM: 'entry.85265664=',
      ISSUEID_PARAM: '&entry.2137816233=',
      MSG: 'Are you satisfied with the resolution of your issue?',
    }
  }

};
module.exports = CONSTANT_VALUES;