# Local data

```text
data/raw/         Immutable source files
data/interim/     Audited intermediate transformations
data/processed/   Versioned development/final-test splits
```

- Không commit dataset và không sửa raw file tại chỗ.
- Với mỗi nguồn phải ghi source, permission/license, received date, schema, size và
  SHA-256 trước khi xử lý.
- Employee ID chỉ dùng cho lineage và split validation, không dùng làm model feature.
- Không đặt CSV giả hoặc dataset mẫu trong repository.
