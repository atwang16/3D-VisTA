import json
from abc import ABC, abstractmethod

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from pipeline.registry import registry, setup_imports
from pipeline.pipeline import Pipeline


class GroundingInferencePipeline(Pipeline):
    def __init__(self, cfg):
        # build saver and logger
        self.saver = registry.get_utils(cfg["saver"]["name"])(**cfg["saver"]["args"])

        # build model
        self.lang_encoder = registry.get_language_model(cfg["lang_encoder"]["name"])(
            **cfg["lang_encoder"]["args"]
        ).cuda()
        self.point_encoder = registry.get_vision_model(cfg["point_encoder"]["name"])(
            **cfg["point_encoder"]["args"]
        ).cuda()
        self.unified_encoder = registry.get_vision_model(cfg["unified_encoder"]["name"])(
            **cfg["unified_encoder"]["args"]
        ).cuda()
        self.ground_head = registry.get_other_model(cfg["ground_head"]["name"])(**cfg["ground_head"]["args"]).cuda()
        self.qa_head = registry.get_other_model(cfg["qa_head"]["name"])(**cfg["qa_head"]["args"]).cuda()
        self.pretrain_head = registry.get_other_model(cfg["pretrain_head"]["name"])(
            **cfg["pretrain_head"]["args"]
        ).cuda()
        self.caption_head = registry.get_other_model(cfg["caption_head"]["name"])(**cfg["caption_head"]["args"]).cuda()

        # build dataset
        self.test_dataset = registry.get_dataset(cfg["refer_dataset"]["name"])(**cfg["refer_dataset"]["args"])

        # build dataloader
        self.batch_size = cfg["batch_size"]
        self.test_data_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=8, pin_memory=True, shuffle=False
        )

        # restore model
        if cfg["restore_model"]:
            self.restore_model()

        self.output_path = cfg.output

    def prepare_data(self, data_dict):
        for key in data_dict:
            if torch.is_tensor(data_dict[key]):
                data_dict[key] = data_dict[key].cuda()

    def initialize(self):
        pass

    def run(self):
        self.set_model_state("eval")
        eval_results = []

        # run
        for i, data_dict in enumerate(tqdm(self.test_data_loader)):
            # forward
            data_dict = self.forward_one(data_dict)

            # save object info
            og3d_pred = torch.argmax(data_dict["og3d_logits"], dim=1)
            item_ids = data_dict["data_idx"]
            for i in range(len(item_ids)):
                centroids_extents_refer = data_dict["obj_boxes"][i][og3d_pred[i]].cpu().numpy().tolist()
                eval_results.append(
                    {
                        "prompt_id": item_ids[i],
                        "scene_id": data_dict["scene_id"][i],
                        "prompt": data_dict["sentence"][i],
                        "predicted_boxes": [centroids_extents_refer[:3], centroids_extents_refer[3:6]],
                    }
                )

        # save results
        with open(self.output_path, "w") as fp:
            json.dump(eval_results, fp, indent=2)

    def forward_one(self, data_dict):
        # prepare data
        self.prepare_data(data_dict)

        # basic feature extracter
        lang_basic_features = self.lang_encoder(data_dict["txt_ids"], data_dict["txt_masks"]).last_hidden_state
        point_basic_features, point_features_pre, obj_cls_raw_logits = self.point_encoder(
            data_dict["obj_fts"].float(),
            data_dict["obj_locs"],
            data_dict["obj_masks"],
            data_dict["obj_sem_masks"],
        )

        # unifed language entity transformer
        language_fuse_feature, point_fuse_feature = self.unified_encoder(
            lang_basic_features,
            data_dict["txt_masks"],
            point_basic_features,
            data_dict["obj_locs"],
            data_dict["obj_masks"],
        )

        # task head
        txt_cls_logits, obj_cls_post_logits, obj_cls_pre_logits, og3d_logits = self.ground_head(
            language_fuse_feature, point_fuse_feature, point_features_pre, data_dict["obj_masks"]
        )
        answer_scores = self.qa_head(
            point_fuse_feature, data_dict["obj_masks"], language_fuse_feature, data_dict["txt_masks"]
        )
        txt_lm_cls_logits, scene_txt_match_logit = self.pretrain_head(language_fuse_feature)
        txt_caption_cls_logit = self.caption_head(language_fuse_feature)

        data_dict["txt_cls_logits"] = txt_cls_logits
        data_dict["obj_cls_post_logits"] = obj_cls_post_logits
        data_dict["obj_cls_pre_logits"] = obj_cls_pre_logits
        data_dict["obj_cls_raw_logits"] = obj_cls_raw_logits
        data_dict["og3d_logits"] = og3d_logits
        data_dict["answer_scores"] = answer_scores
        data_dict["txt_lm_cls_logits"] = txt_lm_cls_logits
        data_dict["scene_txt_match_logit"] = scene_txt_match_logit
        data_dict["txt_caption_cls_logit"] = txt_caption_cls_logit

        return data_dict

    def restore_model(self):
        state_dict = self.saver.restore_dict()
        self.lang_encoder.load_state_dict(state_dict["lang_encoder"])
        self.point_encoder.load_state_dict(state_dict["point_encoder"])
        self.unified_encoder.load_state_dict(state_dict["unified_encoder"])
        self.ground_head.load_state_dict(state_dict["ground_head"])
        try:
            self.qa_head.load_state_dict(state_dict["qa_head"])
        except:
            print("fail to load qa params")
        self.pretrain_head.load_state_dict(state_dict["pretrain_head"])
        try:
            self.caption_head.load_state_dict(state_dict["caption_head"])
        except:
            print("fail to load caption params")

    def set_model_state(self, state="train"):
        assert state in ["train", "eval"]
        torch.cuda.empty_cache()
        if state == "train":
            self.lang_encoder.train()
            self.point_encoder.train()
            self.unified_encoder.train()
            self.ground_head.train()
            self.qa_head.train()
            self.pretrain_head.train()
            self.caption_head.train()
        else:
            self.lang_encoder.eval()
            self.point_encoder.eval()
            self.unified_encoder.eval()
            self.ground_head.eval()
            self.qa_head.eval()
            self.pretrain_head.eval()
            self.caption_head.eval()

    def end(self):
        pass


@hydra.main(version_base=None, config_path="project/vista", config_name="inference_config")
def main(cfg: DictConfig):
    setup_imports()
    pipeline = GroundingInferencePipeline(cfg.pipeline)
    pipeline.run_all()


if __name__ == "__main__":
    main()
