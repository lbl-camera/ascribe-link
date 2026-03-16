# Specimens Directory

Each subdirectory is a specimen bundle containing:

```
specimens/
  brain/
    specimen.json       # required — metadata
    thumbnail.png       # thumbnail image (png, jpg, webp)
    brain.stl           # data file (stl, obj, fbx, bin, etc.)
```

## specimen.json format

```json
{
    "id": "brain",
    "display_name": "Brain",
    "description": "Human brain model",
    "type": "mesh",
    "data_file": "brain.stl",
    "thumbnail_file": "thumbnail.png",
    "story_text": ["Narrative text shown in the viewer."],
    "tags": ["anatomy", "mesh"]
}
```

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `id` | No | Unique identifier (defaults to directory name) |
| `display_name` | Yes | Human-readable name shown in menus |
| `description` | No | Short description for listings |
| `type` | No | `"mesh"` or `"volume"` (default: `"mesh"`) |
| `data_file` | Yes | Filename of the specimen data in this directory |
| `thumbnail_file` | No | Filename of thumbnail (or auto-detected as `thumbnail.*`) |
| `story_text` | No | Array of narrative text paragraphs |
| `tags` | No | Array of string tags for filtering |
